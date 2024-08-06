# Breast Cancer Classification with K-Nearest Neighbors
This project demonstrates how to classify breast cancer as malignant or benign using the K-Nearest Neighbors (KNN) algorithm. The dataset used is the well-known Breast Cancer Wisconsin dataset, which is included in the scikit-learn library.

## Overview
The code in this project performs the following tasks:
1. Loading and exploring the dataset: Understands the features and target labels of the Breast Cancer dataset.
2. Preprocessing the data: Splits the data into training and testing sets.
3. Model training: Utilizes the K-Nearest Neighbors algorithm to build a classification model.
4. Model evaluation: Assesses the model's performance using accuracy and confusion matrix metrics.
5. Model saving: Saves the trained model for future use.

## Prerequisites
* Google Colab: All the code was run in Google Colab, which provides a free cloud-based environment for running Python code.
* Python Libraries: The following Python libraries are required:
    * 'matplotlib'
    * 'joblib'
    * 'pandas'
    * 'numpy'
    * 'scikit-learn'
These libraries are pre-installed in Google Colab, so no additional setup is required.

## How to Run the Code
1. Clone or download the repository: You can clone this repository or download the code files to your local machine.
2. Upload to Google Colab:
     * Open Google Colab (https://colab.research.google.com/).
     * Upload the Python script (.py file) or notebook (.ipynb file) to Colab.
3. Run the Code:
     * Execute each cell in the Colab notebook, or run the script cell-by-cell if using a script.
     * The dataset is loaded from the scikit-learn library, so there's no need to manually download it.
     * The model will be trained, evaluated, and saved as cancer-classifier.dmp.

## key Concepts
  * K-Nearest Neighbors (KNN): A simple, non-parametric, and instance-based learning algorithm used for classification tasks.
  * Breast Cancer Dataset: A dataset containing features computed from digitized images of breast mass, used to predict whether a mass is benign or malignant.

## Code Explanation
The project is organized into the following main steps:
1. Data Loading: Loads the breast cancer dataset using scikit-learn's datasets.load_breast_cancer() function.
2. Data Exploration: Prints feature names, target labels, and provides a statistical summary of the dataset using pandas.
3. Data Splitting: Splits the dataset into training and testing sets using train_test_split from scikit-learn.
4. Model Training: Trains the KNN model on the training data.
5. Model Evaluation: Predicts the test set and evaluates the model's performance using accuracy score and confusion matrix.
6. Model Saving: Saves the trained model using joblib.dump for later use.

## Results
The model achieves a certain level of accuracy (as determined by running the code in Google Colab) and can be further improved or tested on different datasets.

## Future Work
* Experiment with different values of k in KNN to see if it improves accuracy.
* Explore other machine learning algorithms and compare their performance.
* Apply cross-validation to ensure the model's robustness.










