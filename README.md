# Fatty Liver Disease Prediction

This project demonstrates the process of building and evaluating machine learning models to predict whether a person has fatty liver disease based on a set of features including Age, Gender, BMI, and liver enzyme levels (ALT, AST).

## Dataset

A synthetic dataset is generated for testing purposes with the following columns:

- **Age**: Age of the individual (between 20 and 70).
- **Gender**: Gender of the individual (Male/Female).
- **BMI**: Body Mass Index (between 18 and 35).
- **ALT**: Alanine aminotransferase levels (between 10 and 50).
- **AST**: Aspartate aminotransferase levels (between 10 and 40).
- **target**: The label (0 for No fatty liver, 1 for Fatty liver).

The dataset is saved as a CSV file: `fatty_liver_data.csv`.

## Steps

### 1. Data Preprocessing:
- **Categorical Encoding**: Gender column is one-hot encoded using `pandas.get_dummies()`.
- **Handling Missing Values**: Missing numeric values are imputed with the mean of the respective columns.
- **Feature Scaling**: Feature scaling is performed using `StandardScaler` to normalize the features.

### 2. Handling Class Imbalance:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: SMOTE is applied to address the class imbalance problem by generating synthetic samples for the minority class.

### 3. Model Training:
The following models are trained and tuned:

- **Random Forest Classifier**: Hyperparameter tuning is done using `GridSearchCV` to find the best parameters (e.g., number of estimators, max depth, min samples split).
- **Support Vector Machine (SVM)**: A basic SVM model is trained on the dataset.
- **Logistic Regression**: A logistic regression model is also trained for comparison.

### 4. Model Evaluation:
Each model is evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Classification Report**: Precision, Recall, F1-score for both classes (0 and 1).
- **Confusion Matrix**: Shows the true positive, false positive, true negative, and false negative values.

The best model is selected based on accuracy.

## Requirements

To run this project, you need to install the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`

You can install the dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
