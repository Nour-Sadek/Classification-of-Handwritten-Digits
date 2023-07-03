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
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV


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
{max_score}\n')


"""Stage 4: Data preprocessing

Description

At this stage, we will improve model performance by preprocessing the data. We 
will see how normalization affects the accuracy.

Objectives

1 - Import sklearn.preprocessing.Normalizer transformer
2 - Initialize the normalizer and transform the features (x_train and x_test)
3 - Re-run the function <fit_predict_eval> but on the normalized data
4 - Answer the questions:
    - Does the normalization have a positive impact in general?
    - Which two models show the best scores?

"""

# initialize the normalizer and transform the features x_train and x_test
train_transformer = Normalizer().fit(x_train)
x_train_norm = train_transformer.transform(x_train)

test_transformer = Normalizer().fit(x_test)
x_test_norm = test_transformer.transform(x_test)

print("Re-running the models but on the normalized data instead:\n")
# Re-run the function on each of the four models but with the normalized data
normalized_scores = []
for model in models:
    model_score = fit_predict_eval(model(), x_train_norm, x_test_norm, y_train,
                                 y_test)
    normalized_scores.append(model_score)

print(f'The answer to the 1st question: yes\n')

# To help find the best scores
temp_copy = normalized_scores[:]
temp_copy.sort(reverse=True)

# The model that showed the best score
max_normalized_score = temp_copy[0]
index_max_normalized_score = normalized_scores.index(max_normalized_score)

# The model that showed the second best score
second_normalized_scored = temp_copy[1]
index_second_normalized_score = normalized_scores.index(second_normalized_scored)

print(f'The answer to the 2nd question: \
{models[index_max_normalized_score].__name__}-{max_normalized_score}, \
{models[index_second_normalized_score].__name__}-{second_normalized_scored}')


"""Stage 5: Hyperparameter tuning

Description

In the final stage, you need to improve model performance by tuning the 
hyperparameters. No need to do it manually, as sklearn has convenient tools for 
this task. We'll be using GridSearchCV.
We'll focus on only improving the performance of K-nearest Neighbors and 
Random Forest.

Objectives

1 - Initialize GridSearchCV(estimator=..., param_grid=..., scoring='accuracy',
n_jobs=-1) to search over the following parameters:
    - For the K-nearest Neighbors classifier: n_neighbors = [3, 4], weights = 
    ['uniform', 'distance'], algorithm = ['auto', 'brute']
    - For the Random Forest classifier: n_estimators = [300, 500], 
    max_features = ['auto', 'log2'], class_weight = 
    ['balanced', 'balanced_subsample'], criterion = ['gini', 'entropy', 
    'log_loss']
2 - Run the fit method for GridSearchCV. This might take a while...
3 - Print the best sets of parameters for both algorithms.
4 - Train two best estimators on the test set and print their accuracies.

"""

# For K-NearestNeighbor
knn_model = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 4],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'brute']}
knn_grid = GridSearchCV(knn_model, knn_param_grid, scoring='accuracy',
                        n_jobs=-1)

# For RandomForest
rf_model = RandomForestClassifier(random_state=40)
rf_param_grid = {'n_estimators': [300, 500],
                 'max_features': ['sqrt', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}
rf_grid = GridSearchCV(rf_model, rf_param_grid, scoring='accuracy', n_jobs=-1)

# Running the fit method for GridSearchCV
knn_grid.fit(x_train_norm, y_train)
rf_grid.fit(x_train_norm, y_train)

# Getting the best parameters
knn_best = knn_grid.best_params_
rf_best = rf_grid.best_params_

# Getting the accuracy using the best parameters for each model
# for the KNN model
knn_model_ = KNeighborsClassifier(algorithm=knn_best['algorithm'],
                                  n_neighbors=knn_best['n_neighbors'],
                                  weights=knn_best['weights'])
knn_model_.fit(x_train_norm, y_train)
knn_prediction = knn_model_.predict(x_test)
knn_score = accuracy_score(y_test, knn_prediction)
knn_score = round(knn_score, 4)
# for the rf model
rf_model_ = RandomForestClassifier(class_weight=rf_best['class_weight'],
                                   max_features=rf_best['max_features'],
                                   n_estimators=rf_best['n_estimators'])
rf_model_.fit(x_train_norm, y_train)
rf_prediction = rf_model_.predict(x_test)
rf_score = accuracy_score(y_test, rf_prediction)
rf_score = round(rf_score, 4)

print(f"""K-nearest neighbours algorithm
best estimator: KNeighborsClassifier(algorithm={knn_best['algorithm']}, \
n_neighbors={knn_best['n_neighbors']}, weights={knn_best['weights']})
accuracy: {knn_score}\n
""")

print(f"""Random forest algorithm
best estimator: RandomForestClassifier(class_weight={rf_best['class_weight']}, \
max_features={rf_best['max_features']}, n_estimators={rf_best['n_estimators']},\
 random_state=40)
accuracy: {rf_score}
""")
