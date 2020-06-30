"""
Linear regression model with Stochastic Gradient Descent.
"""
import numpy as np
from sklearn.metrics import r2_score

class LinearRegress:

    """
    Linear Regression with Gradient Descent.
    """

    def __init__(
        self,
        bias=True,
        learning_rate=0.01,
        max_iteration=1000,
        keep_hist=True,
        batch_size=10,
    ):
        """
        Linear Regression with Gradient Descent.
        
        params
        ------
        bias: bool. Default=True. Whether to add bias variable or not.
        
        learning_rate: float. learning rate for gradient descent. 
        
        max_iteration: scalar. Maximum iteration for Gradient Descent.
        
        keep_hist: bool. If true, stores J and W for each iteration.
        
        batch_size: int. batch size for each iteration.  
        
        """
        self.bias = bias
        self.w = None
        self.error = None
        self.alpha = learning_rate
        self.max_iter = max_iteration
        self.n_samples = None
        self.n_features = None
        self.keep_hist = keep_hist
        self.history = []
        self.batch_size = batch_size
    def gradient_descent(self, X, y):
        """
        Simple SGD. 

        params
        ------
        X: Predictors. Shape(n_samples, n_features)

        y: Target. Shape(n_samples)

        """
        n_samples = len(y)

        # inserts column of 1s for bias.
        X = np.insert(X, 0, 1, axis=1)

        # Weight Matrix. Shape(n_features,1)
        w = np.random.rand(X.shape[1])

        for itr in range(self.max_iter):
            idxs = np.random.randint(0,n_samples,self.batch_size)
            tmpX = X[idxs]
            tmpy = y[idxs]
            # forward pass
            yhat = np.dot(tmpX, w)  # (self.batch_size, 1)
            # objective
            error = yhat - tmpy
            J = (1 / (2 * self.batch_size)) * np.dot(error, error)

            # backward pass
            w = w - (self.alpha / self.batch_size) * np.dot(tmpX.T, error)

            if self.keep_hist is True:
                self.history.append((J, w))
            if self.verbose and (itr % 100 == 0 or itr == self.max_iter - 1):
                print("Iteration: %d J: %.2f" % (itr, J))
        self.w = w

    def fit(self, X, y, verbose=True):
        """
        Train linear regression model.
        """
        self.verbose = verbose
        self.gradient_descent(X, y)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        params
        ------
        X: array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead, shape = (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples used in the fitting for the estimator.
            
        y: array-like of shape (n_samples,)
            True values for X.
        
        returns
        -------
        score: float.
            R^2 score.
        """
        yhat = self.predict(X)
        return r2_score(y, yhat)

    def predict(self, X):
        """
        Predicts using linear model.
        
        params
        ------
        X: array-like of shape (n_samples, n_features)
            Test samples.
        returns
        -------
        y: array-like of shape(n_samples,)
            Output of the model.
        """
        X = np.insert(X, 0, 1, axis=1)
        yhat = np.dot(X, self.w)
        return yhat
