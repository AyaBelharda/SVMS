Overview
This repository contains code for training a Support Vector Machine (SVM) classifier on the Iris dataset using scikit-learn. The SVM is implemented with a linear kernel to classify Iris flowers into two classes based on sepal length and width.

Files
svm_iris_classification.py: Python script containing the SVM training, evaluation, and visualization.
README.md: This file.
Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Installation
Clone the repository:
bash
Copier le code
git clone <repository-url>
cd <repository-directory>
Install dependencies:
Copier le code
pip install -r requirements.txt
Usage
Run the script svm_iris_classification.py:

Copier le code
python svm_iris_classification.py
The script loads the Iris dataset, preprocesses it by selecting two features (sepal length and width) and two classes (Setosa and Versicolor), splits the data into training and testing sets, and standardizes the features.

It trains a linear SVM model using the training data and evaluates its performance on the test set by calculating accuracy, printing a classification report, and displaying a confusion matrix.

Additionally, the script visualizes the decision boundary and support vectors of the SVM model using matplotlib and seaborn. It also performs cross-validation to assess the model's generalization performance.

Outputs
Accuracy: Reported as a percentage.
Classification Report: Provides precision, recall, F1-score, and support for each class.
Confusion Matrix: Visual representation showing the number of correct and incorrect predictions.
Decision Boundary Plot: Displays the SVM's decision boundary and support vectors in a scatter plot.
Contact
For questions or issues regarding this repository, please contact la_belharda@esi.dz.



