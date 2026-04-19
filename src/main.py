# =========================================================
# LINEAR CLASSIFIERS PROJECT - FROM SCRATCH
# Improved version to better cover the full assignment:
# - Perceptron One-vs-All (OVA)
# - Perceptron One-vs-One (OVO)
# - Softmax Regression
# - Author: Grigoris Liatsos
# =========================================================

import os
import csv
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist


# =========================================================
# REPRODUCIBILITY
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# CONFIG
# =========================================================
RESULTS_DIR = "results_project2_full"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 10
BALANCED_SAMPLES_PER_CLASS = 1000
IMBALANCED_MAIN_CLASS = 0
IMBALANCED_MAIN_SAMPLES = 1000
IMBALANCED_OTHER_SAMPLES = 50

VALIDATION_RATIO = 0.2

# Hyperparameter grids
OVA_LRS = [0.001, 0.01, 0.1]
OVA_EPOCHS_LIST = [5, 10, 20]

OVO_LRS = [0.001, 0.01, 0.1]
OVO_EPOCHS_LIST = [3, 5, 10]

SOFTMAX_LRS = [0.0005, 0.001, 0.003]
SOFTMAX_EPOCHS_LIST = [100, 200, 300]

SHOW_PLOTS = False
SAVE_PLOTS = True


# =========================================================
# CLASS NAMES
# =========================================================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================================================
# LOAD DATASET
# =========================================================
(X_train_full_raw, y_train_full), (X_test_raw, y_test) = fashion_mnist.load_data()

# Keep original images for visualization
X_test_images = X_test_raw.copy()

# Flatten + normalize
X_train_full = X_train_full_raw.reshape(X_train_full_raw.shape[0], -1).astype(np.float32) / 255.0
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).astype(np.float32) / 255.0

# =========================================================
# STANDARDIZATION
# =========================================================
mean = np.mean(X_train_full, axis=0)
std = np.std(X_train_full, axis=0) + 1e-8

X_train_full = (X_train_full - mean) / std
X_test = (X_test - mean) / std

# =========================================================
# ADD BIAS FEATURE
# =========================================================
def add_bias_feature(X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

X_train_full = add_bias_feature(X_train_full)
X_test = add_bias_feature(X_test)


# =========================================================
# DATASET CREATION
# =========================================================
def create_balanced_dataset(X, y, samples_per_class=1000):
    X_bal, y_bal = [], []

    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, samples_per_class, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    X_bal = np.vstack(X_bal)
    y_bal = np.hstack(y_bal)

    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def create_imbalanced_dataset(X, y, main_class=0, main_samples=1000, other_samples=50):
    X_imb, y_imb = [], []

    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        if cls == main_class:
            chosen = np.random.choice(idx, main_samples, replace=False)
        else:
            chosen = np.random.choice(idx, other_samples, replace=False)

        X_imb.append(X[chosen])
        y_imb.append(y[chosen])

    X_imb = np.vstack(X_imb)
    y_imb = np.hstack(y_imb)

    perm = np.random.permutation(len(y_imb))
    return X_imb[perm], y_imb[perm]


def stratified_train_val_split(X, y, val_ratio=0.2):
    X_train_parts, y_train_parts = [], []
    X_val_parts, y_val_parts = [], []

    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        idx = np.random.permutation(idx)

        n_val = max(1, int(len(idx) * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_val_parts.append(X[val_idx])
        y_val_parts.append(y[val_idx])

        X_train_parts.append(X[train_idx])
        y_train_parts.append(y[train_idx])

    X_train_split = np.vstack(X_train_parts)
    y_train_split = np.hstack(y_train_parts)
    X_val_split = np.vstack(X_val_parts)
    y_val_split = np.hstack(y_val_parts)

    train_perm = np.random.permutation(len(y_train_split))
    val_perm = np.random.permutation(len(y_val_split))

    return (
        X_train_split[train_perm], y_train_split[train_perm],
        X_val_split[val_perm], y_val_split[val_perm]
    )


# =========================================================
# METRICS
# =========================================================
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_manual(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(y_true, y_pred, num_classes=10):
    results = {}
    for cls in range(num_classes):
        idx = np.where(y_true == cls)[0]
        cls_acc = np.mean(y_pred[idx] == y_true[idx]) if len(idx) > 0 else 0.0
        results[cls] = cls_acc
    return results


# =========================================================
# PLOTTING
# =========================================================
def save_or_show_plot(filename=None):
    if SAVE_PLOTS and filename is not None:
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, title, filename=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_or_show_plot(filename)


def plot_training_curve(values, title, ylabel, filename=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    save_or_show_plot(filename)


def plot_examples(X_images, y_true, y_pred, title, correct=True, num_examples=10, filename=None):
    idx = np.where(y_true == y_pred)[0] if correct else np.where(y_true != y_pred)[0]

    if len(idx) == 0:
        print(f"No {'correct' if correct else 'incorrect'} examples found for: {title}")
        return

    chosen = np.random.choice(idx, min(num_examples, len(idx)), replace=False)

    plt.figure(figsize=(15, 6))
    for i, sample_idx in enumerate(chosen):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_images[sample_idx], cmap="gray")
        plt.title(
            f"T: {class_names[y_true[sample_idx]]}\nP: {class_names[y_pred[sample_idx]]}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    save_or_show_plot(filename)


# =========================================================
# PERCEPTRON OVA
# =========================================================
class PerceptronOVA:
    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.loss_history = []
        self.train_accuracy_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.W = np.zeros((NUM_CLASSES, num_features), dtype=np.float32)
        self.loss_history = []
        self.train_accuracy_history = []

        for epoch in range(self.epochs):
            errors = 0
            indices = np.random.permutation(num_samples)

            for i in indices:
                xi = X[i]

                for cls in range(NUM_CLASSES):
                    yi = 1 if y[i] == cls else -1
                    if yi * np.dot(self.W[cls], xi) <= 0:
                        self.W[cls] += self.lr * yi * xi
                        errors += 1

            preds_train = self.predict(X)
            train_acc = accuracy(y, preds_train)

            self.loss_history.append(errors)
            self.train_accuracy_history.append(train_acc)

    def predict(self, X):
        scores = X @ self.W.T
        return np.argmax(scores, axis=1)


# =========================================================
# PERCEPTRON OVO
# =========================================================
class PerceptronOVO:
    def __init__(self, lr=0.01, epochs=5):
        self.lr = lr
        self.epochs = epochs
        self.classifiers = {}
        self.pairs = []
        self.loss_history = []
        self.train_accuracy_history = []

    def fit(self, X, y):
        self.classifiers = {}
        self.pairs = []
        self.loss_history = []
        self.train_accuracy_history = []

        classes = np.arange(NUM_CLASSES)

        # Prepare pair datasets once
        pair_data = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1, c2 = classes[i], classes[j]
                idx = np.where((y == c1) | (y == c2))[0]
                X_pair = X[idx]
                y_pair = y[idx]
                y_bin = np.where(y_pair == c1, 1, -1)
                w = np.zeros(X.shape[1], dtype=np.float32)

                self.pairs.append((c1, c2))
                pair_data.append((c1, c2, X_pair, y_bin, w))

        # Global epoch loop so we can keep curves
        for epoch in range(self.epochs):
            total_errors = 0

            new_pair_data = []
            for (c1, c2, X_pair, y_bin, w) in pair_data:
                indices = np.random.permutation(len(y_bin))

                for k in indices:
                    xi = X_pair[k]
                    yi = y_bin[k]
                    if yi * np.dot(w, xi) <= 0:
                        w += self.lr * yi * xi
                        total_errors += 1

                new_pair_data.append((c1, c2, X_pair, y_bin, w))

            pair_data = new_pair_data

            # Store classifiers after each global epoch
            self.classifiers = {(c1, c2): w for (c1, c2, _, _, w) in pair_data}

            preds_train = self.predict(X)
            train_acc = accuracy(y, preds_train)

            self.loss_history.append(total_errors)
            self.train_accuracy_history.append(train_acc)

    def predict(self, X):
        preds = []

        for xi in X:
            votes = np.zeros(NUM_CLASSES, dtype=int)

            for (c1, c2), w in self.classifiers.items():
                score = np.dot(w, xi)
                if score > 0:
                    votes[c1] += 1
                else:
                    votes[c2] += 1

            preds.append(np.argmax(votes))

        return np.array(preds)


# =========================================================
# SOFTMAX CLASSIFIER
# =========================================================
class SoftmaxClassifier:
    def __init__(self, lr=0.01, epochs=100, reg=1e-4, use_class_weights=False):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.use_class_weights = use_class_weights
        self.W = None
        self.loss_history = []
        self.train_accuracy_history = []

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, C):
        y_one = np.zeros((len(y), C), dtype=np.float32)
        y_one[np.arange(len(y)), y] = 1.0
        return y_one

    def compute_class_weights(self, y, C):
        counts = np.bincount(y, minlength=C)
        weights = len(y) / (C * counts + 1e-12)
        return weights.astype(np.float32)

    def fit(self, X, y):
        n, d = X.shape
        C = NUM_CLASSES

        self.W = np.zeros((d, C), dtype=np.float32)
        y_one = self.one_hot(y, C)

        # class weights
        if self.use_class_weights:
            class_weights = self.compute_class_weights(y, C)
            sample_weights = class_weights[y]
        else:
            sample_weights = np.ones(n, dtype=np.float32)

        self.loss_history = []
        self.train_accuracy_history = []

        for epoch in range(self.epochs):
            # SHUFFLE κάθε epoch
            indices = np.random.permutation(n)
            X_epoch = X[indices]
            y_one_epoch = y_one[indices]
            sample_weights_epoch = sample_weights[indices]

            # Forward
            scores = X_epoch @ self.W
            probs = self.softmax(scores)

            # Loss (cross-entropy + L2 regularization)
            loss = -np.sum(
                sample_weights_epoch *
                np.sum(y_one_epoch * np.log(probs + 1e-12), axis=1)
            ) / np.sum(sample_weights_epoch)
            loss += self.reg * np.sum(self.W * self.W)

            # Gradient
            grad = (X_epoch.T @ ((probs - y_one_epoch) * sample_weights_epoch[:, None])) / np.sum(sample_weights_epoch)
            grad += 2 * self.reg * self.W

            # Update
            self.W -= self.lr * grad

            # Metrics
            preds_train = self.predict(X)
            train_acc = accuracy(y, preds_train)

            self.loss_history.append(loss)
            self.train_accuracy_history.append(train_acc)

    def predict(self, X):
        scores = X @ self.W
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

# =========================================================
# UTILITIES
# =========================================================
def print_per_class_accuracy_table(per_class_acc):
    print("\nPer-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"{cls} - {class_names[cls]:12s}: {acc:.4f}")


def get_model_class(model_name):
    if model_name == "Perceptron OVA":
        return PerceptronOVA
    if model_name == "Perceptron OVO":
        return PerceptronOVO
    if model_name == "Softmax":
        return SoftmaxClassifier
    raise ValueError(f"Unknown model name: {model_name}")


def get_hyperparameter_grid(model_name):
    if model_name == "Perceptron OVA":
        return OVA_LRS, OVA_EPOCHS_LIST
    if model_name == "Perceptron OVO":
        return OVO_LRS, OVO_EPOCHS_LIST
    if model_name == "Softmax":
        return SOFTMAX_LRS, SOFTMAX_EPOCHS_LIST
    raise ValueError(f"Unknown model name: {model_name}")


def save_grid_search_results(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "scenario", "lr", "epochs", "train_acc", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_final_results_csv(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)

    fieldnames = ["model", "scenario", "best_lr", "best_epochs", "test_accuracy"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            filtered_row = {k: row[k] for k in fieldnames}
            writer.writerow(filtered_row)


# =========================================================
# MODEL SELECTION
# =========================================================
def hyperparameter_search(model_name, scenario_name, X_train, y_train, X_val, y_val):
    ModelClass = get_model_class(model_name)
    lrs, epoch_list = get_hyperparameter_grid(model_name)

    best_val_acc = -1.0
    best_lr = None
    best_epochs = None
    search_rows = []

    print("\n" + "-" * 70)
    print(f"Hyperparameter Search | {model_name} | {scenario_name}")
    print("-" * 70)

    for lr in lrs:
        for epochs in epoch_list:
            if model_name == "Softmax":
                use_weights = (scenario_name == "Imbalanced")
                model = ModelClass(lr=lr, epochs=epochs, use_class_weights=use_weights)
            else:
                model = ModelClass(lr=lr, epochs=epochs)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_acc = accuracy(y_train, train_pred)
            val_acc = accuracy(y_val, val_pred)

            print(
                f"{model_name:15s} | {scenario_name:10s} | "
                f"lr={lr:<6} | epochs={epochs:<3d} | "
                f"train={train_acc:.4f} | val={val_acc:.4f}"
            )

            search_rows.append({
                "model": model_name,
                "scenario": scenario_name,
                "lr": lr,
                "epochs": epochs,
                "train_acc": train_acc,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_epochs = epochs

    return best_lr, best_epochs, best_val_acc, search_rows


# =========================================================
# FINAL EVALUATION
# =========================================================
def evaluate_best_model(
    model_name,
    scenario_name,
    X_train_used,
    y_train_used,
    X_test,
    y_test,
    X_test_images,
    best_lr,
    best_epochs
):
    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION | {model_name} | {scenario_name}")
    print("=" * 70)
    print(f"Best hyperparameters: lr={best_lr}, epochs={best_epochs}")

    ModelClass = get_model_class(model_name)

    if model_name == "Softmax":
        use_weights = (scenario_name == "Imbalanced")
        model = ModelClass(
            lr=best_lr,
            epochs=best_epochs,
            use_class_weights=use_weights
        )
    else:
        model = ModelClass(lr=best_lr, epochs=best_epochs)
    model.fit(X_train_used, y_train_used)

    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix_manual(y_test, y_pred, NUM_CLASSES)
    pca = per_class_accuracy(y_test, y_pred, NUM_CLASSES)

    print(f"Test Accuracy: {acc:.4f}")
    print_per_class_accuracy_table(pca)

    base_name = f"{model_name}_{scenario_name}".replace(" ", "_")

    plot_confusion_matrix(
        cm,
        title=f"{model_name} - {scenario_name}",
        filename=f"{base_name}_confusion_matrix.png"
    )

    if hasattr(model, "loss_history") and len(model.loss_history) > 0:
        y_label = "Errors" if "Perceptron" in model_name else "Loss"
        plot_training_curve(
            model.loss_history,
            title=f"{model_name} - {scenario_name} Training Curve",
            ylabel=y_label,
            filename=f"{base_name}_training_curve.png"
        )

    if hasattr(model, "train_accuracy_history") and len(model.train_accuracy_history) > 0:
        plot_training_curve(
            model.train_accuracy_history,
            title=f"{model_name} - {scenario_name} Train Accuracy",
            ylabel="Accuracy",
            filename=f"{base_name}_train_accuracy.png"
        )

    plot_examples(
        X_test_images,
        y_test,
        y_pred,
        title=f"{model_name} - {scenario_name} | Correct Classifications",
        correct=True,
        num_examples=10,
        filename=f"{base_name}_correct_examples.png"
    )

    plot_examples(
        X_test_images,
        y_test,
        y_pred,
        title=f"{model_name} - {scenario_name} | Incorrect Classifications",
        correct=False,
        num_examples=10,
        filename=f"{base_name}_incorrect_examples.png"
    )

    return {
        "model": model_name,
        "scenario": scenario_name,
        "best_lr": best_lr,
        "best_epochs": best_epochs,
        "test_accuracy": acc,
        "per_class_accuracy": pca,
        "confusion_matrix": cm
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("Creating balanced and imbalanced datasets...")

    X_bal, y_bal = create_balanced_dataset(
        X_train_full, y_train_full,
        samples_per_class=BALANCED_SAMPLES_PER_CLASS
    )

    X_imb, y_imb = create_imbalanced_dataset(
        X_train_full, y_train_full,
        main_class=IMBALANCED_MAIN_CLASS,
        main_samples=IMBALANCED_MAIN_SAMPLES,
        other_samples=IMBALANCED_OTHER_SAMPLES
    )

    print("\nBalanced distribution:")
    print(Counter(y_bal))

    print("\nImbalanced distribution:")
    print(Counter(y_imb))
    print(f"\nMain class in imbalanced scenario: {IMBALANCED_MAIN_CLASS} ({class_names[IMBALANCED_MAIN_CLASS]})")

    scenarios = [
        ("Balanced", X_bal, y_bal),
        ("Imbalanced", X_imb, y_imb)
    ]

    model_names = [
        "Perceptron OVA",
        "Perceptron OVO",
        "Softmax"
    ]

    all_search_rows = []
    final_results = []

    for scenario_name, X_scenario, y_scenario in scenarios:
        X_train_split, y_train_split, X_val_split, y_val_split = stratified_train_val_split(
            X_scenario, y_scenario, val_ratio=VALIDATION_RATIO
        )

        for model_name in model_names:
            best_lr, best_epochs, best_val_acc, search_rows = hyperparameter_search(
                model_name=model_name,
                scenario_name=scenario_name,
                X_train=X_train_split,
                y_train=y_train_split,
                X_val=X_val_split,
                y_val=y_val_split
            )

            all_search_rows.extend(search_rows)

            print(
                f"\nBest for {model_name} | {scenario_name}: "
                f"lr={best_lr}, epochs={best_epochs}, val_acc={best_val_acc:.4f}"
            )

            # Retrain on full scenario training data with best hyperparameters
            final_result = evaluate_best_model(
                model_name=model_name,
                scenario_name=scenario_name,
                X_train_used=X_scenario,
                y_train_used=y_scenario,
                X_test=X_test,
                y_test=y_test,
                X_test_images=X_test_images,
                best_lr=best_lr,
                best_epochs=best_epochs
            )

            final_results.append(final_result)

    # Save grid search results
    save_grid_search_results(all_search_rows, "grid_search_results.csv")

    # Save final results CSV
    save_final_results_csv(final_results, "final_results.csv")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for res in final_results:
        print(
            f"{res['model']:15s} | {res['scenario']:10s} | "
            f"lr={res['best_lr']:<6} | epochs={res['best_epochs']:<3d} | "
            f"Test Accuracy = {res['test_accuracy']:.4f}"
        )

    summary_path = os.path.join(RESULTS_DIR, "summary_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FINAL SUMMARY\n")
        f.write("=" * 70 + "\n")
        for res in final_results:
            f.write(
                f"{res['model']:15s} | {res['scenario']:10s} | "
                f"lr={res['best_lr']:<6} | epochs={res['best_epochs']:<3d} | "
                f"Test Accuracy = {res['test_accuracy']:.4f}\n"
            )

    print(f"\nAll results saved in folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()