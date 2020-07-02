class GaussNB:
    def __init__(self):
        """
        No params are needed for basic functionality.
        """
        pass
    
    def _mean(self,X): # CHECKED
        """
        Returns class probability for each 
        """
        mu = dict()
        for i in self.classes_:
            idx = np.argwhere(self.y == i).flatten()
            mean = []
            for j in range(self.n_feats):
                mean.append(np.mean( X[idx,j] ))
            mu[i] = mean
        return mu
    
    def _stddev(self,X): # CHECKED
        sigma = dict()
        for i in self.classes_:
            idx = np.argwhere(self.y==i).flatten()
            stddev = []
            for j in range(self.n_feats):
                stddev.append( np.std(X[idx,j]) )
            sigma[i] = stddev
        return sigma
    
    def _prior(self): # CHECKED
        """Prior probability, P(y) for each y
        """
        P = {}
        for i in self.classes_:
            count = np.argwhere(self.y==i).flatten().shape[0]
            probability = count / self.y.shape[0]
            P[i] = probability
        return P
    
    def _normal(self,x,mean,stddev): # CHECKED
        """
        Gaussian Normal Distribution
        $P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$
        """
        
        multiplier = (1/ float(np.sqrt(2 * np.pi * stddev**2))) 
        exp = np.exp(-((x - mean)**2 / float(2 * stddev**2)))
        return multiplier * exp

    
    def P_E_H(self,x,h):
        """
        Uses Normal Distribution to get, P(E|H) = P(E1|H) * P(E2|H) .. * P(En|H)
        
        params
        ------
        X: 1dim array. 
            E in P(E|H)
        H: class in y
        """
        pdfs = []
        
        for i in range(self.n_feats):
            mu = self.means_[h][i]
            sigma = self.stddevs_[h][i]
            pdfs.append( self._normal(x[i],mu,sigma) )
            
        p_e_h = np.prod(pdfs)
        return p_e_h
        
        
    def fit(self, X, y):
        self.n_samples, self.n_feats = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = np.unique(y)
        self.y = y
        
        self.means_ = self._mean(X) # dict of list {class:feats}
        self.stddevs_ = self._stddev(X) # dict of list {class:feat}
        self.priors_ = self._prior() # dict of priors 
        
    def predict(self,X):
        samples, feats = X.shape
        result = []
        for i in range(samples):
            distinct_likelyhoods = []
            for h in self.classes_:
                tmp = self.P_E_H(X[i],h)
                distinct_likelyhoods.append( tmp * self.priors_[h])
            marginal = np.sum(distinct_likelyhoods)
            tmp = 0
            probas = []
            for h in self.classes_:
                numerator = self.priors_[h] * distinct_likelyhoods[tmp]
                denominator = marginal
                probas.append( numerator / denominator )
                tmp+=1
            # predicting maximum
            idx = np.argmax(probas)
            result.append(self.classes_[idx])
        return result
