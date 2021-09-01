# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <a href='https://ai.meng.duke.edu'> = <img align="left" style="padding-top:10px;" src=https://storage.googleapis.com/aipi_datasets/Duke-AIPI-Logo.png>

# # Model Selection & Evaluation

# In this notebook we are going to look at strategies to divide your dataset in order to perform model selection and testing using subsets of data in ways that do not create bias in your measurement of model performance.
#
# We are going to use a dataset which comes from a study done to try to use sonar signals to differentiate between a mine (simulated using a metal cylinder) and a rock.  Details on the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))

# +
# Import the libraries we know we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")
# -

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
data = pd.read_csv(url, header=None)
print(data.shape)
data.head()

# We can see that we have 208 observations (sonar readings), and each observation has 60 features (energy in a particular frequency band summed over a set period of time) and a target value (rock 'R' or mine 'M').  
#
# Let's do one more thing right now, which is to set up an instance of our model.  We will use a Multi-layer Perceptron classifier (a simple form of neural network).  Don't worry about the details now, we will learn them later.  For now, you can treat this model as a black box.

# Create an instance of the MLPClassifier algorithm and set the hyperparameter values
model = MLPClassifier(hidden_layer_sizes=(100,50,10),activation='tanh',
                      solver='sgd',learning_rate_init=0.001,max_iter=2000, random_state=0)

# ## Part 1: Training and test sets
# In this part, you should complete the following:  
# - Split your data into a feature matrix X and a target vector y  
# - Split the data into a training set and a test set, using 85% of the data for training and 15% for testing (hint: use scikit-learn's train_test_split() method, already imported for you.  Name the resulting arrays `X_train, y_train, X_test, y_test`
# - Train (fit) your model on the X and y training sets  
# - Use your trained model to get predictions on the `X_test` test set, and name the predictions `preds`  
# - Finally, run the next code cell to calculate the display the accuracy of your classifier model

# +
### YOUR CODE HERE ###






### END CODE ###
# -

# Evaluate the performance of our model using the test predictions
acc_test = sum(preds==y_test)/len(y_test)
print('Accuracy of our classifier on the test set is {:.3f}'.format(acc_test))

# ## Part 2: Model selection using validation sets
# But what if we want to compare different models (for example, evaluate different algorithms or fine-tune our hyperparameters)?  Can we use the same strategy of training each model on the training data and then comparing their performance on the test set to select the best model?
#
# When we are seeking to optimize models by tuning hyperparameters or comparing different algorithms, it is a best practice to do so by comparing the performance of your model options using a "validation" set, and then reserve use of the test set to evaluate the performance of the final model you have selected.  To utilize this approach we must split our data three ways to create a training set, validation set, and test set.
#
# To illustrate this, let's compare two different models, which are defined for your below

# +
# Create an instance of each model we want to evaluate

model1 = MLPClassifier(hidden_layer_sizes=(100,50,10),activation='tanh',
                      solver='sgd',learning_rate_init=0.001,max_iter=2000, random_state=0)

model2 = MLPClassifier(hidden_layer_sizes=(100,50),activation='relu',
                      solver='sgd',learning_rate_init=0.01,max_iter=2000, random_state=0)
# -

# In this part you should complete the following:  
# - Split your X and y arrays into a training set and a test set, using 15% of data for the test set.  Store the training data as `X_train_full, y_train_full` and the test set data as `X_test, y_test`
# - Now, split your training set again into a training set and a validation set, using 15% of the training set for the new validation set (and the remaining 85% is still available for training). Store the final training data as `X_train, y_train` and the validation set data as `X_val, y_val`
# - Train (fit) model1 and model2 using the training data only  
# - Now, use your trained model1 and model2 to generate predictions on the validation set.  Store model1's predictions as `val_preds_model1` and model2's predictions as `val_preds_model2`  
# - Finally, run the code cell below to calculate the accuracy of each on the validation set.  Based on this, which model would you select as your final model?

# +
### YOUR CODE HERE ###





### END CODE ###
# -

# Now let's compare two different models and determine which one gives us better performance.

# +
# Calculate the validation accuracy of each model
acc_val_model1 = sum(val_preds_model1==y_val)/len(y_val)
acc_val_model2 = sum(val_preds_model2==y_val)/len(y_val)

print('Accuracy of model1 on the validation set is {:.3f}'.format(acc_val_model1))
print('Accuracy of model2 on the validation set is {:.3f}'.format(acc_val_model2))
# -

# Now that we've chosen our final model, we can use the test set to evaluate it's performance.  Before we do that, let's retrain our model using the training plus validation data.

# +
# Train our selected model on the training plus validation sets
model2.fit(X_train_full,y_train_full)

# Evaluate its performance on the test set
preds_test = model2.predict(X_test)
acc_test = sum(preds_test==y_test)/len(y_test)
print('Accuracy of our model on the test set is {:.3f}'.format(acc_test))
# -

# ## Part 3: Model selection using cross-validation

# A common approach to comparing and optimizing models is to use cross-validation rather than a single validation set to compare model performace.  We will then select the better model based on the cross-validation performance and use the test set to determine its performance.

# +
# Let's set aside a test set and use the remainder for training and cross-validation
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,test_size=0.15)

# Set up the two models we want to compare: a neural network model and a KNN model
model_a = MLPClassifier(hidden_layer_sizes=(100,50),activation='relu',
                      solver='sgd',learning_rate_init=0.01,max_iter=1000,random_state=0)

model_b = KNeighborsClassifier(n_neighbors=5)

### YOUR CODE HERE ###

# Instantiate the KFold generator which allows you to iterate through the data k times, splitting the data
# into the training folds and validation fold each time


# For each model, use K-folds cross validation to calculate the cross-validation accuracy

    
    # For each iteration of the cross-validation
    
        # Train the model on the training folds
        
        # Calculate the accuracy on the validation folds
    
        
    # Calculate the mean validation accuracy across all iterations
    
    
### END CODE ###
# -

# As we can see above, the cross-validation accuracy of model_a is higher than model_b, so we will use model_a.  Let's now evaluate the performance of model_a on the test set

# +
# Train our selected model on the full training set
model_a.fit(X_train,y_train)
    
# Evaluate its performance on the test set
preds_test = model_a.predict(X_test)
acc_test = sum(preds_test==y_test)/len(y_test)
print('Accuracy of our model on the test set is {:.3f}'.format(acc_test))
