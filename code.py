# Imported packages
import tensorflow as tf
import numpy as np

"""Stage 1: The Keras dataset

Description

We start by feeding data to our program. We will use the MNIST digits dataset 
from Keras. 

Objectives

1 - Load the data in your program. We need x_train and y_train only. Skip 
X_test and y_test in return of load_data()
2 - Reshape the features array to the 2D array with n rows (n = number of images 
in the dataset) and m columns (m = number of pixels in each image)
3 - Print information about the dataset: target classes' names; the shape of 
the features array; the shape of the target array; the minimum and maximum 
values of the features array.

"""

# Load the dataset
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Reshape the features array to a 2D array from a 3D array
reshaped_X_train = X_train.reshape(X_train.shape[0],
                                   X_train.shape[1] * X_train.shape[2])

# Print the information about the dataset
print(f"""Classes: {np.unique(y_train)}
Features' shape: {reshaped_X_train.shape}
Target's shape: {y_train.shape}
min: {reshaped_X_train.min()}, max: {reshaped_X_train.max()}
""")
