# NN-from-scratch

## Description
This repository contains a logistic regression classifier that recognizes cats. The code implements a logistic regression model using gradient descent optimization to classify images as either cat or non-cat. The model is trained on a dataset consisting of cat and non-cat images, and its performance is evaluated on both the training and test sets. The code also includes a function to plot the learning curve, showing the cost function over iterations.

## Dataset
The code uses a dataset of cat and non-cat images. The dataset is stored in two separate files: `train_catvnoncat.h5` (training set) and `test_catvnoncat.h5` (test set). Each file contains a matrix of flattened image data and corresponding labels.

## Results
After training the logistic regression model, the following accuracy scores were achieved:
- Training accuracy: 99.04%
- Test accuracy: 70.00%

## Usage
To run the code, follow these steps:
1. Ensure that the required dependencies are installed (NumPy, matplotlib, h5py, scipy, and PIL).
2. Clone the repository and navigate to the project directory.
3. Execute the `main()` function in the `logistic_regression.py` file.
4. The code will load the training and test datasets, preprocess the images, train the logistic regression model, and print the accuracy on both the training and test sets. It will also display a plot showing the learning curve.
