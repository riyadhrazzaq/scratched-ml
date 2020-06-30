"""
Lasso Regressor implementation using numpy.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

plt.style.use('seaborn-white')

# dataset
dbts = datasets.load_diabetes()
X = dbts.data
y = dbts.target.reshape(-1,1)

def soft_threshold(rho,lmbda):
    if rho < -lmbda:
        return rho+lmbda
    elif rho <= lmbda:
        return 0
    else:
        return rho -lmbda

def coord_descent():
    m,n = X.shape
    X = 
