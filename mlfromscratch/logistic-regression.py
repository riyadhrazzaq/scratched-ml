import numpy as np

class LogisticReg:
    
    def __init__(self, learning_rate=0.01,max_iteration=100, batch_size=10):
        self.learning_rate = learning_rate
        self.max_iter = max_iteration
        self.batch_size = batch_size
        
    def sigmoid(self,x):
        if isinstance(x, np.ndarray):
            result = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                result[i] = np.exp(x[i]) / (1+ np.exp(x[i]))
            return result
        else:
            return np.exp(x) / (1+ np.exp(x))
    
    def sgd(self, X, y):
        n_samples, n_features = X.shape
        self.betas = np.zeros(n_features) # column vector, parameters
        costs = []
        for it in range(self.max_iter):
            indices = np.random.randint(0,X.shape[0],size=self.batch_size)

#             ----------------------
#                 Forward Pass
#             ----------------------
            prediction = self.sigmoid(np.dot(X[indices, :],self.betas))
            
            error = y[indices] - prediction
#             ----------------------
#                 Backward Pass
#             ----------------------
            cost = (-1 / indices.shape[0]) * (y[indices] @ np.log(prediction) + (1 - y[indices]) @ np.log(1-prediction) )
            gradient = (1 / indices.shape[0]) * (X[indices, :].T @ error)
        
            self.betas = self.betas - (self.learning_rate * -gradient)
            costs.append(cost)
            
            if it % (self.max_iter / 10)==0:
                accuracy = accuracy_score(y[indices],np.round(prediction))
                print(f"iteration: {it}, Cost: {cost}, Accuracy: {accuracy}")
            
        self.history = costs
            
        
    def plot(self):
        fig, ax = plt.subplots(1,1,figsize=(20,10),facecolor='white')
        ax.plot(range(self.max_iter),self.history)
        plt.show()
        
    def fit(self, X, y):
        """
        Fit logistic model using Stochastic Gradient Descent
        """
        print(X.shape)
        X = np.insert(X,0,1,axis=1) # add 1s for matrix multiplication
        
        self.sgd(X,y)
    
    def predict(self, X):
        X = np.insert(X,0,1,axis=1)
        yhat = np.dot(X,self.betas)
        yhat = self.sigmoid(yhat)
        return np.round(yhat)
    
    def score(self, X,y):
        yhat = self.predict(X)
        return accuracy_score(y,yhat)

