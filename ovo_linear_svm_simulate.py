"""
This module implements linear support vector machine in a one-vs-one classification fashion.
This code includes preprocessing of any given data, function for objective function, 
function for gradient calculation, function for fast gradient descent with backtracking rule
for the step size adjustments and finally a master function that calls all above function
and classifies using a one-vs-one fashion where the mode of each classifier is selected 
as the final prediction.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def preprocess(data_dir):
    """
    Using a given data folder directory, load the training, validation and test set 
    and standardize the data.

    Input:
    :data_dir: folder name where the raw data is stored.

    Return: Processed version of X_train, y_train, X_val, y_val and X_test

    """
    # Load the data
    x_train = np.load(os.path.join(data_dir, 'train_features.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    x_val = np.load(os.path.join(data_dir, 'val_features.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))
    x_test = np.load(os.path.join(data_dir, 'test_features.npy'))

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_val = scaler.transform(x_val)

 
    return X_train,y_train,X_val,y_val

def obj(beta, lambd, x, y, h=0.5):
    """
    Computes the objective of a linear hinge loss function.

    Input:
    :beta: beta classifiers of which objective value is to be calculated
    :lambd: regularization parameter for the penalty term
    :x: observations with features
    :y: labels for the observations
    :h: hinge loss parameter, preset to 0.5

    Return: objective value of given beta, lambda and dataset
    """
    yt = y*x.dot(beta)
    hinge_loss = (1+h-yt)**2/(4*h)*(np.abs(1-yt) <= h) + (1-yt)*(yt < (1-h)) 

    return np.mean(hinge_loss) + lambd*np.dot(beta, beta)

def grad(beta, lambd, x, y, h=0.5):
    """
    Computes the gradient of a linear hinge loss function.

    Input:
    :beta: beta classifiers of which objective value is to be calculated
    :lambd: regularization parameter for the penalty term
    :x: observations with features
    :y: labels for the observations
    :h: hinge loss parameter, preset to 0.5

    Return: gradient of given beta, lambda and dataset
    """
    yt = y*x.dot(beta)
    hinge_loss_prime = -(1+h-yt)/(2*h)*y*(np.abs(1-yt) <= h) - y*(yt < (1-h)) 
    return np.mean(hinge_loss_prime[:, np.newaxis]*x, axis=0) + 2*lambd*beta

def backtracking(beta, lambd, x, y, step_size, alpha=0.5, frac=0.5, max_iter=100):
    """
    Implement the backtracking line search for fast gradient descent algorithm

    Input:
    :beta: beta classifiers of which objective value is to be calculated
    :lambd: regularization parameter for the penalty term
    :x: observations with features
    :y: labels for the observations
    :step_size: Starting (maximum) step size
    :alpha: Constant used to define sufficient decrease condition
    :frac: Fraction by which we decrease t if the previous t doesn't work
    :max_iter: Maximum number of iterations to run the algorithm

    Return: 
    :step_size: Step size to use
    """
    grad_beta = grad(beta,lambd,x,y)  # Gradient at initial beta
    norm_grad_beta = np.linalg.norm(grad_beta)  # Norm of the gradient at initial beta
    found_t = False
    i = 0  # Iteration counter
    while (found_t is False and i < max_iter):
        # check if the backtracking line search satisfies the norm requirement
        # or stop when max iteration number is reached
        if ((obj(beta - step_size*grad_beta, lambd, x, y)) 
            <= (obj(beta, lambd, x, y) - alpha*step_size*norm_grad_beta**2)):
            found_t = True
        elif i == max_iter - 1:
            raise('Maximum number of iterations of backtracking reached')
        else:
            # update the new step size with the fraction to be reduced
            step_size *= frac
            i += 1
    return step_size

def mylinearsvm(beta, lambd, x, y, step_size_init, eps=0.0000001, max_iter=100):
    """
    Run mylinear svm function with a backtracking step size using fast gradient descent

    Inputs:
    :beta: beta classifiers of which objective value is to be calculated
    :lambd: regularization parameter for the penalty term
    :x: observations with features
    :y: labels for the observations
    :step_size_init: Initial step size (a constant)
    :eps: Value for convergence criterion for the the norm of the gradient.
    :max_iter: Maximum number of iterations to perform

    Output:
    :beta_vals: Matrix of estimated beta's at each iteration,
                with the most recent values in the last row.
    :objs: Matrix of estimated objective vals at each iteration,
                with the most recent values in the last row.      
    """
    theta = beta
    t = step_size_init
    grad_beta = grad(beta, lambd, x, y)
    beta_vals = [beta]
    objs = [obj(beta, lambd, x, y)]
    iter = 0
    while np.linalg.norm(grad_beta) > eps and iter < max_iter: 
        # THE CODE BELOW SO IT USES BACKTRACKING LINE SEARCH INSTEAD OF A CONSTANT STEP SIZE
        t = backtracking(beta, lambd=lambd, x=x, y=y, step_size=t)
        # THE CODE BELOW USES UPDATING THETA FOR BETA OPTIMAZATION
        beta = theta - t*grad_beta
        theta = beta + (iter/(iter+3))*(beta - beta_vals[-1])
        obj_val = obj(beta,lambd, x, y)
        beta_vals.append(beta)
        objs.append(obj_val)
        grad_beta = grad(theta, lambd, x, y)
        iter += 1
        
    return np.array(beta_vals), np.array(objs)

def fold3_CV_for_lambda(x,y,index):
    """
    Three fold cross validation for the optimal lambda value.

    Inputs:
    :x: observations with features
    :y: labels for the observations 
    :index: randomly generated indexing for each fold to be pulled out each time

    Output:
    :avg_error: the error rate for each lambda value tested
    :set_of_lambda: corresponding lambda values that were tested
    """
    # Construct any set of lambda to be tested
    set_of_lambda = np.logspace(-2,5,10)
    # for each lambda, test 3 folds and average error rate 
    # to pick lowest one for corresponding lambda
    avg_error = []
    for lambd_val in set_of_lambda: 
        #loop over 3 lambda values
        error = []
        for n in range(3):
            #calculate average error rate over 3 folds
            beta_vals, obj_vals = mylinearsvm(beta=np.zeros(x.shape[1]),
                                              lambd = lambd_val,
                                              x = x[np.ix_(index != n)],
                                              y = y[np.ix_(index != n)],
                                              step_size_init=1,
                                              max_iter=100)
            
            y_hat = 2*(beta_vals[-1].dot(x[np.ix_(index == n)].T)>0)-1
            
            
            error_rate = np.mean(y_hat != y[np.ix_(index == n)])
            
            error.append(error_rate)
            
        avg_error.append(np.mean(error))
    
    return avg_error, set_of_lambda

def ovo_classifier(X_train, y_train, X_val, y_val):
    """
    Master function that calls all above functions and perform one-vs-one classficiation.

    Inputs:
    :X_train: Observations of training set
    :y_train: Labels of training set
    :X_val: Observations of validation set
    :y_val: Labels of validation set

    Outputs:
    :beta_matrix: beta classifier matrix
    :obj_matrix: objective values matrix
    :y_val_pred: validation labels for misclassification error calculation
    :misclassification_error: error rate on the validation dataset
    """
    beta_matrix = []
    obj_matrix =[]
    final_matrix = []
    final_matrix_test =[]
    for i in np.unique(y_train):
        for j in np.unique(y_train):
            if i < j:
                print(i,j)
                # Slicing X train and y train into each pair wise comparison
                # location scale y labels into [-1,1] for loss calculation
                sliced_X_train = X_train[np.ix_(np.bitwise_or(y_train == i, y_train == j))]
                sliced_y_train = ((y_train[np.ix_(np.bitwise_or(y_train == i, y_train == j))]-min(i,j))/(max(i,j)-min(i,j)))*2-1
                
                # generating index for cross validation indexing
                index = np.random.randint(low=0,high=3,size=sliced_X_train.shape[0])
                print('starting cross validation...', i, j)
                avg_error, set_of_lambda = fold3_CV_for_lambda(x=sliced_X_train, y=sliced_y_train, index=index)
                optimal_lambda = set_of_lambda[np.argmin(avg_error)]
                print('starting mylinearsvm...', i, j)
                beta_vals, obj_vals = mylinearsvm(beta = np.zeros(sliced_X_train.shape[1]),
                                             	  lambd = optimal_lambda,
                                                  x = sliced_X_train,
                                                  y = sliced_y_train,
                                                  step_size_init=1)
                
                # Store classifier and objective value from each iteration
                beta_matrix.append(beta_vals[-1])
                obj_matrix.append(obj_vals[-1])
                print('starting prediction...', i, j)
                # Predict y_val from X_val using the trained classifiers
                pred = 2*(beta_vals.dot(X_val.T)>0)-1
                y_hat = np.zeros(X_val.shape[0])
                y_hat[pred[-1]==1] = j
                y_hat[pred[-1]==-1] = i
                final_matrix.append(y_hat)


    prediction_matrix = np.array(final_matrix)
    
    y_val_pred = []
    for f in range(X_val.shape[0]):
        data = pd.DataFrame(prediction_matrix[:,f].astype(int))
        # Use mode to find the most frequent class
        assignment = data.mode()
        # In the case of a tie, use random int generator for the indexing of a random mode
        if assignment.shape[0] > 1:
            assign = assignment.loc[np.random.randint(assignment.shape[0]),0]
        else:
            assign = assignment.loc[0,0]
        y_val_pred.append(assign)
        
    misclassification_error = np.mean(np.array(y_val_pred) != y_val)
    print('misclassification_error:', misclassification_error)
        
    return np.array(beta_matrix), np.array(obj_matrix), np.array(y_val_pred), misclassification_error


def visualization(obj_value):
    """
    Visualize the objective value decrease over iterations.

    Inputs:
    :obj_value: objective values calculated from ovo_classifier

    Return: plot of objective values for each class
    """
    for n in range(1):
        plt.loglog(obj_value[n],".");

    plt.ylabel('objective values');
    plt.xlabel('iteration counter');
    plt.title('objective values for each pair against iterations');
    plt.legend();
    plt.show();


if __name__ == '__main__':
    X_train,y_train,X_val,y_val,X_test = preprocess('data')
    beta, obj, y_val, y_test, error = ovo_classifier(X_train,y_train,X_val,y_val)
    pd.DataFrame(beta).to_csv("beta.csv")
    pd.DataFrame(beta).to_csv("obj.csv")
    pd.DataFrame(y_val).to_csv("y_val.csv")
    pd.DataFrame([error]).to_csv("error.csv")
    visualization(obj)




