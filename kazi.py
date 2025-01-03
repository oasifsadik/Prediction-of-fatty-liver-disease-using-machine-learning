import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

data = {
    'Age': np.random.randint(20, 70, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'BMI': np.random.uniform(18, 35, size=100),
    'ALT': np.random.uniform(10, 50, size=100),
    'AST': np.random.uniform(10, 40, size=100),
    'target': np.random.choice([0, 1], size=100)
}

data_df = pd.DataFrame(data)
data_df.to_csv('fatty_liver_data.csv', index=False)

data = pd.read_csv('fatty_liver_data.csv')

data = pd.get_dummies(data, drop_first=True)
data = data.fillna(data.select_dtypes(include=[np.number]).mean())

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid,
                              cv=3, n_jobs=-1, verbose=2)

rf_grid_search.fit(X_train, y_train)

best_rf_model = rf_grid_search.best_estimator_

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

rf_predictions = best_rf_model.predict(X_test)
print("Random Forest Results (Tuned):")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

svm_predictions = svm_model.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))

logreg_predictions = logreg_model.predict(X_test)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Classification Report:\n", classification_report(y_test, logreg_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, logreg_predictions))

models = {'Random Forest': best_rf_model, 'SVM': svm_model, 'Logistic Regression': logreg_model}
best_model = max(models, key=lambda model: accuracy_score(y_test, models[model].predict(X_test)))
print(f"\nThe best performing model is: {best_model}")

best_model_instance = models[best_model]
