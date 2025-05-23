# Image-Classification-using-CNN-TensorFlow-

COMPANY - CODTECH IT SOLUTIONS

NAME - G.NITHISH KUMAR REDDY

INTERN ID - CT04DN40

DOMAIN - MACHINE LEARNING

DURATION - 4 WEEKS

MENTOR - NEELA SANTHOSH

# 📘 Project Title:
Image Classification Using Convolutional Neural Networks (CNN) with TensorFlow

# 📌 Project Description:
This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras for the purpose of image classification. CNNs are a powerful class of deep learning models particularly suited for computer vision tasks. The model is trained and evaluated on the CIFAR-10 dataset, which contains 60,000 color images in 10 classes, with 6,000 images per class. This project covers all essential components including data preprocessing, model building, training, and evaluation.

# 🎯 Objective:
The objective of this project is to develop a deep learning model that can classify images into their correct categories based on patterns it learns during training. It aims to highlight the practical use of CNNs in recognizing visual patterns and automating the image classification process.

# 📂 Dataset:
The CIFAR-10 dataset is a widely used benchmark in machine learning for image recognition tasks. It contains 10 classes:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Each image is 32x32 pixels with 3 color channels (RGB). The dataset is divided into 50,000 training images and 10,000 testing images.

# 🔧 Methodology:
1. Data Preprocessing:
The image data is normalized to scale pixel values to the range [0, 1].

Labels are reshaped to a suitable format for the loss function used.

2. Model Architecture:
The CNN model consists of multiple layers:

Convolutional layers to extract visual features like edges and textures.

MaxPooling layers to downsample feature maps and reduce spatial dimensions.

Flattening layer to convert the 2D feature maps into a 1D feature vector.

Fully connected (Dense) layers to learn non-linear combinations of features.

Softmax output layer to predict the probability of each class.

3. Compilation:
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Metrics: Accuracy

4. Training:
The model is trained over 10 epochs.

Validation is performed using the test set to monitor generalization performance.

5. Evaluation:
Final performance is evaluated on the test set.

The accuracy and loss are recorded.

A training/validation accuracy plot is generated to visualize the learning process.

# ✅ Results:
The trained CNN model achieves significant accuracy on the test dataset, demonstrating its ability to generalize and correctly classify unseen images. Accuracy improves with each epoch and converges to a stable point, indicating successful training. Visualization of predictions and accuracy curves provides insight into the model’s learning behavior.

![image](https://github.com/user-attachments/assets/0e5a8b1d-6cb9-47f7-a67b-9f6fbe47ccde)


