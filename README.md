# Image Classification using Convolutional Neural Networks (CNN)
Authored by saeed asle
# Description
This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification.
The dataset used is the Intel Image Classification dataset, which consists of images belonging to six categories:
buildings, forest, glacier, mountain, sea, and street.

# steps: 
* Data Loading and Preprocessing: Loads the dataset and preprocesses the images for training, testing, and prediction.
* Model Creation: Defines a CNN model using Keras with various convolutional and pooling layers followed by dense layers.
* Model Training: Trains the CNN model on the training data to learn the features of different image categories.
* Model Evaluation: Evaluates the trained model on the test data to measure its performance in terms of loss and accuracy.
* Prediction: Uses the trained model to make predictions on new images from the prediction dataset.
# Features
* Data Loading and Preprocessing: Loads images from the dataset, resizes them to a specified size, and prepares them for training, testing, and prediction.
* CNN Model Creation: Defines a CNN model using Keras with convolutional, pooling, and dense layers.
* Model Training: Trains the CNN model on the training data to learn the features of different image categories.
* Model Evaluation: Evaluates the trained model on the test data to measure its performance in terms of loss and accuracy.
* Prediction: Uses the trained model to make predictions on new images from the prediction dataset.
# Dependencies
* pandas: For data manipulation and analysis.
* numpy: For numerical operations.
* matplotlib: For plotting graphs.
* seaborn: For visualization.
* os: For interacting with the operating system.
* glob: For file operations.
* cv2: For image processing.
* tensorflow: For building and training the CNN model.
* keras: For defining the CNN model architecture and training.
# How to Use
* Ensure you have the necessary libraries installed, such as pandas, numpy, matplotlib, seaborn, os, glob, cv2, tensorflow, and keras.
* Download the Intel Image Classification dataset from the provided link and extract it to a directory.
* Update the trainpath, testpath, and predpath variables in the code to point to the respective directories containing the training, testing, and prediction images.
* Run the provided code to load, preprocess, train, test, and predict using the CNN model.
# Output
The code outputs various results and visualizations, including:
* Number of images found in each category for training, testing, and prediction.
* Distribution of image sizes in the dataset.
* Sample images with their corresponding categories for visualization.
* Model summary and details.
* Training progress and metrics (loss, accuracy).
* Test loss and accuracy.
* Predictions on new images with their corresponding categories.
# Note
Ensure that you have sufficient computational resources (CPU/GPU) to train the CNN model, as training deep learning models can be computationally intensive.
