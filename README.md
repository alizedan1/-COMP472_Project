CIFAR-10 Image Classification Project
Contents
• data preprocessing.py: Script to load and preprocess the CIFAR-10
dataset. Includes resizing, normalization, and feature extraction using
ResNet-18 and PCA for dimensionality reduction.
• naive bayes.py: Implementation of Gaussian Naive Bayes classifiers (cus-
tom and Scikit-learn versions).
• decision tree.py: Implementation of decision tree classifiers using both
a custom implementation and Scikit-learn.
• mlp.py: Implementation of a Multi-Layer Perceptron (MLP) for image
classification using PyTorch.
• cnn.py: Implementation of a convolutional neural network (CNN) using
the VGG11 architecture.
• evaluation.py: Script for evaluating the models using accuracy, confu-
sion matrices, precision, recall, and F1-score.
• saved models/: Directory containing trained model files for evaluation
without retraining.
• README.md: This file, providing an overview of the project, setup instruc-
tions, and execution steps.
• report.pdf: Comprehensive project report detailing methodologies, ex-
periments, and findings.
Instructions for Running the Code
Prerequisites
1. Install Python 3.9 or later.
2. Install required Python libraries:
pip install numpy torch torchvision scikit-learn matplotlib
1
1. Data Preprocessing
1. Run the data preprocessing.py script to preprocess the CIFAR-10 dataset:
python data_preprocessing.py
This script will:
• Load CIFAR-10 data and select the first 500 training and 100 test
images per class.
• Resize and normalize images for ResNet-18.
• Extract features using a pre-trained ResNet-18 (removing the last
layer).
• Reduce feature dimensions using PCA.
Output: A saved file containing preprocessed features for training and
testing.
2. Training the Models
Each model script trains the corresponding model and saves the trained model
to the saved models/ directory.
• Naive Bayes:
python naive_bayes.py
• Decision Tree:
python decision_tree.py
• Multi-Layer Perceptron (MLP):
python mlp.py
• Convolutional Neural Network (CNN):
python cnn.py
2
3. Evaluating the Models
Run the evaluation.py script to evaluate the saved models:
python evaluation.py
The script generates accuracy scores, confusion matrices, and other evalu-
ation metrics for each model. Results will be saved as plots and tables in the
results/ directory.
4. Applying the Models
To classify a new image, use the feature extraction pipeline in data preprocessing.py,
then load a specific model from saved models/ and apply it. For example:
from sklearn.externals import joblib
model = joblib.load(’saved_models/naive_bayes.pkl’)
prediction = model.predict(new_image_features)
Notes
• All scripts are modular and can be adapted for experimentation.
• Refer to report.pdf for detailed explanations of the model architectures,
training methods, and performance analysis.
• Ensure you use the same preprocessing steps for any additional datasets
from sklearn.externals import joblib

model = joblib.load('saved_models/naive_bayes.pkl')
prediction = model.predict(new_image_features) 
