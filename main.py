import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    # Calculate the cost
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A)) / m

    # A - Y is the result of dL/dz
    # dL/dw = dL/dz * dz/dw = (A - Y) * dz/dw
    # for wi we have dz/dwi = (x1w1 + x2*w2 + ... + xi*wi + ... + xm*wm + b)' with respect of wi => dL/dwi = xi
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        costs.append(cost)

        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {
        "w": w,
        "b": b
              }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5) * 1.0
    
    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # Initialize parameters with zeros
    m = X_train.shape[0]
    w, b = initialize_with_zeros(m)

    # Gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b 
    w = params["w"]
    b = params["b"]

    # Predict test/train 
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    

def main():
   
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # Reshape the images from (num_px, num_px, 3) to (num_opx * num_px * 3, 1)
    # An image is represented by a column and each row is a feature of the image
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    # For normalization we can divide each pixel by its maximum value(255)
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    # Create the model
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

if __name__ == "__main__":
    main()