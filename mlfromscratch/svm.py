"""
Implementation of Support Vector Machine from scratch. Quadratic Problem was solved using *cvxopt*
"""


class SVC:
    """
    Support Vector Classifier. Only for binary classification with linear kernel.
    """
    def __init__(self):
        self.svX = None
        self.svy = None
        self.b = None 
        self.w = None
    

    def fit(self, X, y):
        tempy = np.outer(y, y)
        tempX = np.inner(X, X)  # replace this for other kernels
        H = tempy * tempX

        # quadratic problem solver
        P = cvxopt.matrix(H, tc="d")
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc="d")
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        solve = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solve["x"])

        # calculate w on all vectors
        self.w = np.zeros(n_features)
        for i in range(n_samples):
            self.w += alphas[i] * y[i] * X[i]

        # set of support vectors
        idxs = alphas > 1e-5
        self.svX = X[idxs]
        self.svy = y[idxs]
        svalphas = alphas[idxs]

        # calculate b from average of SV
        total = 0
        for s in range(svX.shape[0]):
            temp_b = y[s]
            for m in range(svX.shape[0]):
                temp_b -= svalphas[m] * svy[m] * np.inner(svX[m], svX[s])
            total += temp_b
        self.b = temp_b / float(len(idxs))

    def predict(self, X):
        x = np.array([[2, 2]])
        yhat = np.sign(np.inner(self.w, x) + self.b)
        print(yhat)


model = SVC()
model.fit(X, y)
model.predict([[1, 2]])
