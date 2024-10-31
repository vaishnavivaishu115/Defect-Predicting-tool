# Import necessary libraries
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from preprocess import load_and_preprocess_data  # Ensure this function is in preprocess.py

# Load and preprocess the data
data = load_and_preprocess_data('data/defects.csv')  # Ensure your data path is correct

# Check class distribution in the target variable
print("Class distribution in 'is_defective':")
print(data['is_defective'].value_counts())

# Select features and target variable
X = data[['lines_of_code', 'code_churn', 'num_developers', 'past_defects']]
y = data['is_defective']

# Handle class imbalance with SMOTE, setting n_neighbors=1 for very small classes
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=1))
print("Recall:", recall_score(y_test, y_pred, zero_division=1))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=1))

# Perform cross-validation with cv=2 to ensure there are enough samples
scores = cross_val_score(model, X_resampled, y_resampled, cv=2, scoring='f1')
print("Cross-validated F1 scores:", scores)
print("Average F1 Score from Cross-validation:", scores.mean())

# Save the trained model
joblib.dump(model, 'models/defect_model.pkl')
print("Model saved as 'defect_model.pkl' in the models folder.")
