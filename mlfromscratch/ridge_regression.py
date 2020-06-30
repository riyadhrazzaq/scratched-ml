"""
Ridge regression. 
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class RidgeRegressor:
    """
    This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    """

    def __init__(self, lmbda=1, fit_intercept=True, standardize=True,normalize=True):
        """
        params
        ------
        lmbda: float. default=1.0
             regularization parameter. 

        fit_intercept: bool. default=True
            Whether to fit intercept for this model or not. If X,y is not normalized then True is recommended. 

        normalize: bool. default=False.

        standardize: bool. default=True.
        """
        self.standardize = standardize
        self.lmbda = lmbda
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        if self.standardize:
            self.scalar = StandardScaler(with_mean=False)
        self.normalize = normalize
    def _ridge(self, X, y):

        I = np.identity(X.shape[1])
        tmp = np.dot(X.T, X) + (self.lmbda * I)
        self.w = np.dot(np.dot(np.linalg.inv(tmp), X.T), y)

    def fit(self, X, y):
        if self.standardize:
            X = self.scalar.fit_transform(X)
        if self.fit_intercept:
            X = np.insert(X,0,1,axis=1)
        if self.normalize:
            X = normalize(X)
        self._ridge(X, y)

    def predict(self, X):
        if self.standardize:
            X = self.scalar.fit_transform(X)
        if self.fit_intercept:
            X = np.insert(X,0,1,axis=1)
        if self.normalize:
            X = normalize(X)
        y = np.dot(X, self.w)
        return y

    def score(self,X,y):
        return r2_score(y,self.predict(X))
