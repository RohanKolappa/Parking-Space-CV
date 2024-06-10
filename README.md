# Parking-Space-CV

This project explores traditional computer vision techniques and machine learning models for developing a parking space detection system. The goal is to classify parking spaces as occupied or vacant in real-time from video footage. 

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Challenges and Improvements](#challenges-and-improvements)
- [Contributions](#contributions)
- [Conclusion](#conclusion)
- [References](#references)
- [Code Files and Their Roles](#code-files-and-their-roles)

## Introduction

Efficient parking management is crucial in urban areas due to the rise in vehicles and limited parking spaces. Current methods often use costly hardware and can be inaccurate. This project aims to create a low-cost, robust parking space detection system using computer vision. We explore traditional computer vision techniques and machine learning models to find an accurate and efficient real-time solution. Effective parking management reduces congestion, search time, fuel consumption, and emissions, improving user experience and minimizing environmental impact. Our project offers a scalable solution for existing parking infrastructure.

## Datasets

- **PKLot Dataset:** Contains 12,416 images of parking lots extracted from surveillance camera frames, annotated with bounding boxes indicating whether the parking spot is occupied or not.
- **Custom Parking Dataset:** Created using a live video feed, cropped to obtain 69 x 30 images of occupied and vacant spots. This dataset contains 1605 images of occupied spaces and 400 images of vacant spaces.
- **Live Video Parking Lot Feed:** 1920 x 1080 bird’s eye view feed of a parking lot.

## Methodology

### PKLot Dataset
- **Preprocessing:**
  - Auto-orientation of pixel data (EXIF-orientation stripping).
  - Resize to 640x640 (Stretch).

- **Model Architecture:**
  - 2 Convolution Layers using ReLU activation.
  - 2 Pooling Layers.
  - 2 Fully Connected Layers and an output layer.

- **Hyperparameters:**
  - **Epochs:** 10
  - **Batch Size:** 10
  - **Optimizer:** Adam
  - **Loss:** Categorical Cross Entropy
  - **Metrics:** Accuracy

### Custom Parking Dataset
- **Data Preprocessing and Augmentation:**
  - Normalization, horizontal flipping, shear transformation, random zooming.
  - 20% of the dataset set aside for validation.

- **Model Architecture:**
  - 3 Convolutional layers using ReLU activation.
  - 2 Max Pooling layers.
  - 1 Flatten layer.
  - 1 Dense layer using ReLU.
  - 1 50% Dropout layer.
  - 1 Output dense layer with a single neuron using sigmoid activation.

- **Hyperparameters:**
  - **Epochs:** 50
  - **Batch Size:** 32
  - **Optimizer:** Adam
  - **Loss:** Binary Cross Entropy
  - **Metrics:** Accuracy

### Traditional Computer Vision Approach
- **Preprocessing:**
  - Convert frames to grayscale.
  - Apply adaptive thresholding to segment the parking lot into binary regions.

- **Adaptive Threshold Hyperparameters:**
  - **Adaptive Method:** Gaussian
  - **Block Size:** 53
  - **C:** 30
  - **Threshold Type:** Binary Inverse

- **Occupancy Detection:**
  - Calculate the number of white pixels in predefined bounding boxes.
  - Classify spaces as occupied if there are more than 250 white pixels.

## Experimental Results

- **PKLot Dataset:** Test Accuracy: 99.84%
- **Custom Cropped Parking Dataset:** Test Accuracy: 99.74%
- **Traditional Computer Vision Approach:** Provided real-time performance and accurate detection for larger parking lots.

## Challenges and Improvements

- **Challenges Faced:**
  - Lighting variations and occlusions.
  - Inefficiencies in prediction time for larger parking lots using CNN.

- **Improvements Over Baseline Methods:**
  - Data augmentation techniques improved model robustness.
  - Traditional computer vision approach provided faster and reliable detection.

## Contributions

- **Athul Krishna Sughosh:** ML Model using PKLot dataset, Report
- **Brian Dao:** ML Model on CCTV video and Traditional CV Approach, Report
- **Rohan Kolappa:** Model Evaluation, Slides, Report

## Conclusion

Despite the CNN's high accuracy, its prediction time per space was impractical for larger parking lots. The traditional computer vision approach was chosen for its efficiency and real-time performance, providing a reliable solution for automated parking management. Future work includes incorporating more diverse datasets and enhancing model robustness to function under different conditions.

## References

- **Datasets:**
  - PKLot Dataset: https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset
  - Custom Parking Dataset: https://www.kaggle.com/datasets/mfaisalqureshi/parking
  - Live Video Parking Lot Feed: https://drive.google.com/drive/folders/1CjEFWihRqTLNUnYRwHXxGAVwSXF2k8QC
 


## Code Files and Their Roles

- **main.py**
  - **Purpose:** Main script for processing video input to detect parking spaces.
  - **Functionality:** Reads video frames, converts them to grayscale, applies adaptive thresholding, and uses the trained model to classify each parking space as occupied or empty. The results are displayed in real-time.

- **ParkingLotConfigure.py**
  - **Purpose:** Script for setting up and saving the coordinates of parking spaces.
  - **Functionality:** Allows users to define the locations of regular and disabled parking spaces on a static image. The coordinates are saved in a pickle file for later use.

- **ParkingLotConfigure2.py**
  - **Purpose:** An alternative version of the parking lot configuration script.
  - **Functionality:** Similar to ParkingLotConfigure.py, but utilizes a different pickle file.

- **CarClassifier.ipynb**
  - **Purpose:** Jupyter notebook for training the parking space classifier model.
  - **Functionality:** Contains code for loading datasets, preprocessing images, augmenting data, defining the CNN architecture, and training the model.

- **PKLot_model.ipynb**
  - **Purpose:** Jupyter notebook for training the parking space classifier model.
  - **Functionality:** Contains code for loading datasets, defining the CNN architecture, training, and evaluating the model.

- **parking_space_classifier.h5**
  - **Purpose:** Trained model file.
  - **Functionality:** Contains the trained CNN model weights and architecture, used by main.py for classifying parking spaces in the video frames.

- **rectangles.pkl & rectangles2.pkl**
  - **Purpose:** Files storing the coordinates of the parking spaces.
  - **Functionality:** These files store the coordinates of the regular and disabled parking spaces defined using the configuration scripts. They are loaded by main.py during video processing.
