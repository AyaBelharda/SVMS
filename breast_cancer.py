import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Linear SVM ###

# Create and train the Linear SVM model
svm_model_linear = SVC(kernel='linear', random_state=6)
svm_model_linear.fit(X_train_scaled, y_train)

# Make predictions on the test set with Linear SVM
y_pred_linear = svm_model_linear.predict(X_test_scaled)

# Calculate accuracy of the Linear SVM model
accuracy_linear = accuracy_score(y_test, y_pred_linear) * 100
print(f'Accuracy with linear SVM: {accuracy_linear:.2f}%')

# Display the classification report for Linear SVM
print('Classification Report for Linear SVM:')
print(classification_report(y_test, y_pred_linear, target_names=data.target_names))

# Display the confusion matrix for Linear SVM
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print("Confusion Matrix for Linear SVM:")
print(conf_matrix_linear)


"""
### RBF SVM ###

# Uncomment the following block to use RBF SVM
# Create and train the RBF SVM model
svm_model_rbf = SVC(kernel='rbf', gamma='scale', random_state=6)
svm_model_rbf.fit(X_train_scaled, y_train)

# Make predictions on the test set with RBF SVM
y_pred_rbf = svm_model_rbf.predict(X_test_scaled)

# Calculate accuracy of the RBF SVM model
accuracy_rbf = accuracy_score(y_test, y_pred_rbf) * 100
print(f'Accuracy with RBF SVM: {accuracy_rbf:.2f}%')

# Display the classification report for RBF SVM
print('Classification Report for RBF SVM:')
print(classification_report(y_test, y_pred_rbf, target_names=data.target_names))

# Display the confusion matrix for RBF SVM
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print("Confusion Matrix for RBF SVM:")
print(conf_matrix_rbf)
"""
