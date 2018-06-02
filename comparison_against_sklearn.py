"""
This module calls the ovo_linear_svm and compares to the sklearn version of
SVC with linear kernel to compare performance in terms of misclassification error.
"""
import ovo_linear_svm_real_world
from sklearn import svm

def ovo_sklearn(X_train,y_train,X_val,y_val,X_test):
    """
    sklearn function that fits linear svm kernel.

    Inputs:
    :X_train: Observations of training set
    :y_train: Labels of training set
    :X_val: Observations of validation set
    :y_val: Labels of validation set
    :X_test: Observations of test set

    Outputs:
    :y_test: predicted test labels
    :y_val_hat: validation labels for misclassification error calculation
    :misclassification_error: error rate on the validation dataset
    """
    clf = svm.SVC(C=1.0,kernel='linear')
    clf.fit(X_train, y_train)
            
    y_test = clf.predict(X_test)
    y_val_hat = clf.predict(X_val)
    misclassification_error = np.mean(y_val_hat != y_val)
    print('Sklearn misclassification error is:',misclassification_error )
    
    return y_test, y_val_hat, misclassification_error

if __name__ == '__main__':
   X_train,y_train,X_val,y_val,X_test = ovo_linear_svm_real_world.preprocess('data')
   beta, obj, y_val, y_test, error = ovo_linear_svm_real_world.ovo_classifier(X_train,y_train,X_val,y_val,X_test)
   y_test, y_val, sklearn_error = ovo_sklearn(X_train,y_train,X_val,y_val,X_test)
