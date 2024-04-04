import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Data Preprocessing
# Load Titanic dataset
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
titanic_df = pd.DataFrame(titanic.data, columns=titanic.feature_names)
titanic_df['survived'] = titanic.target

# Separate features & target variable
X = titanic_df.drop(columns=['survived'])
y = titanic_df['survived']

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_train_final, X_test_final = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Hyperparameter Tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2']
}
logreg = LogisticRegression()
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_imputed, y_train)

# Best parameters & accuracy
best_params = grid_search.best_params_

# Retrain logistic regression model
logreg_best = LogisticRegression(**best_params)
logreg_best.fit(X_train_imputed, y_train)

# Evaluate performance
y_pred_best = logreg_best.predict(X_test_imputed)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Accuracy (Best Model):", accuracy_best)
print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best))

# Save model to disk
joblib.dump(logreg_best, 'logistic_regression_model.pkl')

# prediction function
def predict_survival(input_data):
    loaded_model = joblib.load('logistic_regression_model.pkl')
    predictions = loaded_model.predict(input_data)
    return predictions

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_best)
classification_rep = classification_report(y_test, y_pred_best)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print accuracy and classification report
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)
