**Distracted Driver Detection – CNN-based Image Classification
**
**Overview**
This project focuses on detecting driver distractions using a Convolutional Neural Network (CNN) model trained on image data. The goal is to classify driver behavior into 10 distinct categories (e.g., texting, drinking, talking to passengers) using the State Farm Distracted Driver Detection dataset (https://www.kaggle.com/c/state-farm-distracted-driver-detection/data).

**Dataset Summary
**The dataset contains over 22,000 labeled images spread across 10 classes (c0 to c9), representing different distracted driving behaviors. Each image is associated with a subject (i.e., driver ID), enabling both classification and per-user analytics.


**Class	Description
**c0	Safe driving
c1	Texting - right hand
c2	Talking on the phone - right
c3	Texting - left hand
c4	Talking on the phone - left
c5	Operating the radio
c6	Drinking
c7	Reaching behind
c8	Hair and makeup
c9	Talking to passenger

**Project Workflow
**1. Data Loading & Preprocessing
Images are resized to 225x225 and loaded from train/c{0-9}/ folders

Labels are mapped numerically (c0 = 0, c1 = 1, ..., c9 = 9)

Converted images and labels to NumPy arrays

One-hot encoded target labels

Split into 75% training and 25% testing

2. Data Exploration
26 unique drivers

Imbalanced number of images per driver; potential for stratified modeling

Visualized sample images for each class

3. Model Architecture
A custom CNN built using Keras, with the following layers:

3 blocks of Conv2D → ELU → MaxPooling2D → Dropout

Flatten → Dense (256 units) → Dropout

Output layer with softmax activation (10 classes)

**Training Details:
**
Loss Function: Categorical Crossentropy

Optimizer: RMSProp

Regularization: Dropout layers (25–50%)

Callbacks: ModelCheckpoint, EarlyStopping to prevent overfitting

**Final Evaluation
**
Final Test Accuracy: 98.95%

This high accuracy reflects the model’s strong performance in classifying distracted driving behaviors across 10 classes.
