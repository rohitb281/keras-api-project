# 🧠 Loan Repayment Prediction — Keras API Project

A machine learning project that builds a **neural network using the Keras API** to predict whether a borrower will repay a loan. The model is trained on historical lending data and evaluated using classification metrics.

---

## 📌 Overview

This project uses a deep learning classification model built with **Keras (TensorFlow backend)** to predict loan repayment outcomes. The workflow includes data preprocessing, feature scaling, model building, training, and evaluation.

The goal is to demonstrate practical usage of neural networks for real-world financial risk prediction.

---

## 🎯 Objective

Predict whether a borrower will:
- ✅ Repay the loan  
- ❌ Default on the loan  

Using borrower financial and credit-related features.

---

## 🧩 Tech Stack

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📊 Dataset

The dataset contains borrower and loan-related attributes that may include:

- Income
- Credit history
- Loan amount
- Interest rate
- Employment length
- Debt ratio
- Payment history indicators

Target variable:

- loan_repaid (0 or 1)
  
---

## 🔬 Project Workflow

### 1️⃣ Data Preprocessing
- Load dataset
- Handle missing values
- Remove leakage features
- Encode categorical variables
- Feature scaling using StandardScaler / MinMaxScaler

---

### 2️⃣ Exploratory Data Analysis
- Distribution plots
- Correlation analysis
- Class balance check
- Feature relationship visualization

---

### 3️⃣ Model Building (Keras Sequential API)

Neural network includes:
- Dense layers
- Activation functions (ReLU / Sigmoid)
- Dropout for regularization
- Binary classification output layer

Example structure:
- Input Layer
- Dense Hidden Layers
- Dropout Layers
- Output Layer (Sigmoid)


---

### 4️⃣ Training

- Train/Test split
- Model compiled with:
  - Binary crossentropy loss
  - Adam optimizer
- Model trained over multiple epochs
- Validation monitoring

---

### 5️⃣ Evaluation

Model evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Precision / Recall / F1-score

---

## 📈 Results

`classification_report: `
```
precision    recall  f1-score   support

           0       0.99      0.44      0.61     15658
           1       0.88      1.00      0.93     63386

    accuracy                           0.89     79044
   macro avg       0.93      0.72      0.77     79044
weighted avg       0.90      0.89      0.87     79044
```

---

## ▶️ How to Run

### Clone repo
```bash
git clone https://github.com/your-username/keras-api-project.git
cd keras-api-project
```

---

## Install dependencies
```
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## Open the notebook:
```
jupyter notebook
```

Run all cells in:

```
Keras-Project-loan-repayment.ipynb
```

---

## 🧠 Concepts Demonstrated

- Neural network classification
- Keras Sequential API
- Feature preprocessing pipeline
- Overfitting control with Dropout
- Model evaluation metrics
- Binary classification workflow

---

## 🚀 Possible Improvements

- Hyperparameter tuning
- ross-validation
- ROC curve & AUC analysis
- Feature importance analysis
- Model comparison with non-NN algorithms
- Deployment as API

---

## ⚠️ Notes
- Results depend on preprocessing choices
- Class imbalance handling can improve performance
- Feature leakage must be avoided for real deployment

---

## 📄 License
- Open for educational and portfolio use.

---

## 👤 Author
- Rohit Bollapragada
- GitHub: https://github.com/rohitb281
