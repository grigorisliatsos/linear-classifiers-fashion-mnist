# 🧠 Linear Classifiers from Scratch – Fashion-MNIST

## 📌 Overview

This project presents a complete implementation of **linear classifiers from scratch** for multi-class image classification using the **Fashion-MNIST dataset**.

The goal is to study the behavior, strengths, and limitations of linear models under:
- ✅ Balanced data distributions
- ❌ Imbalanced data scenarios

All models are implemented **from scratch using NumPy**, without relying on high-level machine learning libraries such as scikit-learn.

---

## 🚀 Implemented Models

The following models are implemented:

- 🔹 **Perceptron (One-vs-All - OVA)**
- 🔹 **Perceptron (One-vs-One - OVO)**
- 🔹 **Softmax Regression (Multinomial Logistic Regression)**

Each model is built manually, including:
- Training loops
- Weight updates
- Prediction logic
- Evaluation metrics

---

## 📊 Key Results

| Model | Balanced Accuracy | Imbalanced Accuracy |
|------|-----------------|--------------------|
| Perceptron OVA | 78.13% | 67.32% |
| Perceptron OVO | **79.97%** | 71.31% |
| Softmax Regression | 76.36% | **75.17%** |

### 🔍 Key Insights

- 🔹 **Perceptron OVO** performs best in balanced datasets
- 🔹 **Softmax Regression** handles class imbalance more effectively
- 🔹 Linear models struggle with **visually similar classes**
- 🔹 Class imbalance significantly affects model performance

---

## 📊 Dataset

- Dataset: **Fashion-MNIST**
- Total samples: 70,000
- Image size: 28×28 (grayscale)
- Classes (10):
  - T-shirt/top, Trouser, Pullover, Dress, Coat
  - Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## ⚙️ Data Preprocessing

The following preprocessing steps are applied:

- Flatten images (28×28 → 784 features)
- Normalize pixel values to [0, 1]
- Standardization (zero mean, unit variance)
- Add **bias feature manually**

---

## ⚙️ Experimental Setup

### 🟢 Scenario 1: Balanced Dataset
- 1000 samples per class

### 🔴 Scenario 2: Imbalanced Dataset
- 1 dominant class: 1000 samples
- Remaining classes: 50 samples each

---

## 🔍 Model Selection

Hyperparameters are tuned using **grid search with validation split**.

### Example hyperparameter grids:

```python
OVA_LRS = [0.001, 0.01, 0.1]
OVA_EPOCHS = [5, 10, 20]

OVO_LRS = [0.001, 0.01, 0.1]
OVO_EPOCHS = [3, 5, 10]

SOFTMAX_LRS = [0.0005, 0.001, 0.003]
SOFTMAX_EPOCHS = [100, 200, 300]
