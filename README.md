# Deep Learning Approaches for Detecting Pneumonia Cases

## Overview
This project implements deep learning models for automated detection of pneumonia in chest X-ray images. The code explores three approaches:
1. A **Custom Convolutional Neural Network (CNN)** model.
2. A **Pre-Trained MobileNet** model, fine-tuned for pneumonia classification.
3. A **Hierarchical Model**, which first distinguishes between 'Normal' and 'Pneumonia' and then classifies pneumonia as 'Bacterial' or 'Viral'.

The goal of this project is to demonstrate how deep learning can be applied to detect pneumonia, which is crucial for early diagnosis and treatment.

## What the Code Does
1. **Data Loading**: The dataset consists of chest X-ray images, which are preprocessed by resizing, normalization, and augmentation (to balance the data).
2. **Model Training**: Three different models (Custom CNN, MobileNet, and Hierarchical) are trained on the X-ray images to classify them as either 'Normal' or 'Pneumonia.' The hierarchical model further distinguishes between 'Bacterial' and 'Viral' pneumonia cases.
3. **Model Evaluation**: The models are evaluated using standard classification metrics, including accuracy, precision, recall, F1-score, and AUC.
4. **Results**: Performance metrics are displayed for each model, and comparisons are made to identify the best-performing approach.

## How It Works

### 1. Data Preprocessing
The dataset consists of chest X-ray images labeled as 'Normal' or 'Pneumonia'. The preprocessing steps include:
- **Resizing**: All images are resized to a fixed size (e.g., 224x224 pixels) to ensure uniformity.
- **Normalization**: The pixel values are normalized to a range of [0, 1] to improve model performance.
- **Data Augmentation**: Techniques like rotation, flipping, and zooming are applied to the images to help the model generalize better.

### 2. Model Training
#### Custom CNN:
- A custom CNN architecture is designed with multiple convolutional layers followed by pooling and fully connected layers.
- The model is trained on the preprocessed dataset using a categorical cross-entropy loss function and Adam optimizer.

#### Pre-Trained MobileNet:
- MobileNet, a lightweight convolutional neural network pre-trained on the ImageNet dataset, is fine-tuned for the pneumonia classification task.
- The model layers are frozen initially, and only the fully connected layers are trained to adjust to the new dataset.

#### Hierarchical Model:
- The first stage of the model differentiates between 'Normal' and 'Pneumonia' cases.
- The second stage classifies pneumonia into 'Bacterial' or 'Viral'.
- The model uses two separate neural networks for the two stages, combining their results for more accurate classification.

### 3. Model Evaluation
The code evaluates each model on the following metrics:
- **Accuracy**: The percentage of correctly classified instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in actual class.
- **F1-Score**: The weighted average of precision and recall.
- **AUC**: The area under the ROC curve, providing a measure of the model's ability to distinguish between classes.

### 4. Output
The results are displayed as:
- Accuracy, Precision, Recall, F1-Score, and AUC for each model.
- A confusion matrix to visualize the model’s performance in terms of true positives, false positives, true negatives, and false negatives.

## How to Run the Code

### Prerequisites
Ensure you have the following libraries installed:
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
