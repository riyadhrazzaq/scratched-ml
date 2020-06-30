"""
Lasso Regressor implementation using numpy.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

class LassoReg:
    def __init__(self,lmbda,max_iters,intercept=False,verbose=False):
        self.max_iters = max_iters
        self.lmbda = lmbda
        self.verbose = verbose
        self.intercept = intercept
        
    def _soft_threshold(self,rho,lmbda):
        if rho < - lmbda:
            return (rho + lmbda)
        elif rho >  lmbda:
            return (rho - lmbda)
        else:
            return 0
    
    def _coord_descent(self):
        n_samples, n_feats = self.X.shape    
        theta = np.ones((n_feats,1))

        for i in range(self.max_iters):
            
            for j in range(n_feats):
                
                x_j = self.X[:,j].reshape(-1,1)
                pred = np.dot(self.X, theta)
                rho = np.dot(x_j.T, self.y - pred + theta[j]*x_j)
                
                if i == self.max_iters-1 and j == n_feats-1 and self.verbose:
                    print(f'At i{i} j{j}',rho, pred.shape, (theta[j] * x_j).shape)
                    
                theta[j] =  self._soft_threshold(rho, self.lmbda)
                    
        return theta.flatten()
    
    def fit(self,X,y):
        self.X = X / np.linalg.norm(X, axis=0)
        self.y = y
        self.thetas = self._coord_descent()
    
    def predict(self,X):
        X = X / np.linalg.norm(X, axis=0)
        yhat = np.dot(X,self.thetas)
        return yhat


