# Imported packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
""", end='\n\n')


"""Stage 2: Split into sets

Description

At this stage, we need to use sklearn to split the data into train and test 
sets. We will use only a portion of the dataset to process model training 
faster in the next stage.

Objectives

1 - Use the first 6000 rows of the datasets. Set the test set size as 0.3 and 
the random seed of 40
2 - Print new datasets' shapes
3 - Print the proportions of samples per class in the training set to make sure 
our dataset is balanced

"""

# Split the first 6000 rows of the dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(reshaped_X_train[:6000, :],
                                                    y_train[:6000],
                                                    test_size=0.3,
                                                    random_state=40)

# Print the required information
print(f"""x_train shape: {x_train.shape}
x_test shape: {x_test.shape}
y_train shape: {y_train.shape}
y_test shape: {y_test.shape}
Proportion of samples per class in train set:
{pd.DataFrame(y_train).value_counts(normalize=True)} 
""")


"""Stage 3: Train models with default settings

Description

We are ready to train our models. In this stage, we need to find the best 
algorithm that can identify handwritten digits. Refer to the following 
algorithms: K-nearest Neighbors, Decision Tree, Logistic Regression, and 
Random Forest. We need to train these four classifiers with default parameters.
We will also use an accuracy metric to evaluate the models.

Objectives

1 - Implement the function <fit_predict_eval> to make the process of training 
the four models faster. This function will initialize and fit the models, then 
it will make predictions and print out the accuracies.
2 - Print the answer to the question: Which model performs better? 

"""


# Implement the fit_predict_eval function
def fit_predict_eval(model, features_train, features_test, target_train,
                     target_test):
    # fit the model, make a prediction, and calculate the accuracy score
    model_ = model
    model_.fit(features_train, target_train)
    prediction = model_.predict(features_test)
    score = accuracy_score(target_test, prediction)
    score = round(score, 4)

    print(f'Model: {model}\nAccuracy: {score}\n')

    return score


# Run the function on each of the four models
models = [KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression,
          RandomForestClassifier]
scores = []
for model in models:
    model_score = fit_predict_eval(model(), x_train, x_test, y_train, y_test)
    scores.append(model_score)

# Answer the question: Which model performs better?
max_score = max(scores)
index_max_score = scores.index(max_score)

print(f'The answer to the question: {models[index_max_score].__name__} - \
{max_score}')
