Breast Cancer Detection Using SVM Classifiers
This script demonstrates the use of Support Vector Machines (SVMs) for classifying breast cancer tumors based on extracted features from digitized images of fine needle aspirates (FNA).

Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn's load_breast_cancer module. It contains 30 features computed from images of breast mass FNA, including measures like radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

Usage
Dependencies:

Python 3.x
NumPy
Matplotlib
scikit-learn (sklearn)
Setup:

Ensure all dependencies are installed (pip install numpy matplotlib scikit-learn).
Download or clone the script and dataset files.
Running the Script:

Execute the script breast_cancer_svm.py.
By default, the script trains and evaluates a Support Vector Machine with a linear kernel.
You can switch to a radial basis function (RBF) kernel by uncommenting the relevant code block and commenting out the linear SVM section.
Interpreting Results:

The script calculates and displays the accuracy, classification report, and confusion matrix for the chosen SVM model.
Accuracy provides a percentage measure of correct predictions on the test set.
The classification report gives precision, recall, F1-score, and support for each class.
The confusion matrix visualizes true positive, true negative, false positive, and false negative counts.
Example Results
Linear SVM:

Accuracy: 98.25%
Classification Report: Detailed precision, recall, and F1-score for benign and malignant tumors.
Confusion Matrix: Visual representation of predicted versus actual classifications.
RBF SVM:

Accuracy: 97.81%
Classification Report: Similar detailed metrics as the linear SVM.
Confusion Matrix: Reflects the performance of the RBF SVM on the test data.
Notes
Experiment with different SVM kernels (linear, poly, rbf) to find the best model for this dataset.
Adjust the random_state parameter in train_test_split for reproducible results.
Further optimization can be done by tuning SVM hyperparameters like C and gamma.
