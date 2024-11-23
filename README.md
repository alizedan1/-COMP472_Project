# -CIFAR-10 Image Classification Project

This repository contains the implementation for the CIFAR-10 image classification project as part of the COMP472 (Artificial Intelligence) course at Concordia University.– Clearly outline the steps to execute your code for data pre-processing.

Contents
data_preprocessing.py: Script to load and preprocess the CIFAR-10 dataset. Includes resizing, normalization, and feature extraction using ResNet-18 and PCA for dimensionality reduction.
naive_bayes.py: Implementation of Gaussian Naive Bayes classifiers (custom and Scikit-learn versions).
decision_tree.py: Implementation of decision tree classifiers using both a custom implementation and Scikit-learn.
mlp.py: Implementation of a Multi-Layer Perceptron (MLP) for image classification using PyTorch.
cnn.py: Implementation of a convolutional neural network (CNN) using the VGG11 architecture.
evaluation.py: Script for evaluating the models using accuracy, confusion matrices, precision, recall, and F1-score.
saved_models/: Directory containing trained model files for evaluation without retraining.
README.md: This file, providing an overview of the project, setup instructions, and execution steps.
report.pdf: Comprehensive project report detailing methodologies, experiments, and findings.
Instructions for Running the Code
Prerequisites
Install Python 3.9 or later.
Install required Python libraries:
bash
Copy code
pip install numpy torch torchvision scikit-learn matplotlib
1. Data Preprocessing
Run the data_preprocessing.py script to preprocess the CIFAR-10 dataset:

bash
Copy code
python data_preprocessing.py
This script will:

Load CIFAR-10 data and select the first 500 training and 100 test images per class.
Resize and normalize images for ResNet-18.
Extract features using a pre-trained ResNet-18 (removing the last layer).
Reduce feature dimensions using PCA.
Output: A saved file containing preprocessed features for training and testing.

2. Training the Models
Each model script trains the corresponding model and saves the trained model to the saved_models/ directory.

Naive Bayes:
bash
Copy code
python naive_bayes.py
Decision Tree:
bash
Copy code
python decision_tree.py
Multi-Layer Perceptron (MLP):
bash
Copy code
python mlp.py
Convolutional Neural Network (CNN):
bash
Copy code
python cnn.py
3. Evaluating the Models
Run the evaluation.py script to evaluate the saved models:

bash
Copy code
python evaluation.py
The script generates accuracy scores, confusion matrices, and other evaluation metrics for each model. Results will be saved as plots and tables in the results/ directory.

4. Applying the Models
To classify a new image, use the feature extraction pipeline in data_preprocessing.py, then load a specific model from saved_models/ and apply it. For example:

python
Copy code
from sklearn.externals import joblib

model = joblib.load('saved_models/naive_bayes.pkl')
prediction = model.predict(new_image_features) 
