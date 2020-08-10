"""
This module implements AdaBoost.M1 algorithm as described in [1],[2].

# References
[1] Freund, Yoav, and Robert E. Schapire. "A desicion-theoretic generalization of on-line learning and an application to boosting." European conference on computational learning theory. Springer, Berlin, Heidelberg, 1995.
[2] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media, 2009.
"""

class AdaBoost:
    def __init__(self, T):
        """
        AdaBoost constructor with Sklearn's Decision Tree.

        params
        ------
        T: int.
            Number of model/tree to train.
        """
        self.T = T
        self.modelContainer = []
        self.modelAlpha = []
        
    def fit(self, X, y):
        """
        params
        ------
        X,y: typical ndarray input.
        """
        nSamples = X.shape[0]
        self.D = np.full(fill_value=(1/nSamples), shape=(nSamples))
        self.modelContainer = []
        self.modelAlpha = np.ones((self.T))
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for t in range(self.T):
            model = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            model.fit(X, y, sample_weight= self.D)
            predictions = model.predict(X)
            error_indices = y != predictions
            err = np.sum(self.D[error_indices]) / np.sum(self.D)
            
            alpha = np.log((1-err) / err)
            
            self.D[error_indices] = self.D[error_indices] * np.exp(alpha)
            self.modelContainer.append(model)
            self.modelAlpha[t] = alpha
#             storing meta information
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

    def predict(self,X):
        """
        params
        ------
        X: typical ndarray input.
        """
        nSamples = X.shape[0]
        Y = np.zeros((nSamples, self.T))
        for t in range(self.T):
            model = self.modelContainer[t]
            Y[:,t] = model.predict(X) * self.modelAlpha[t]
        y = np.sum(Y, axis=1)
        y = np.sign(y)
        return y
    
    def score(self, X, y):
        """
        Returns sklearn's accuracy_score.
        """
        predicted = self.predict(X)
        return accuracy_score(y, predicted)
        
