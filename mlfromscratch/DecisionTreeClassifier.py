import numpy as np
import pandas as pd
class Node:
    def __init__(self,depth=None,attribute=None,value=None,data=None,child=[]):
        self.depth = depth
        self.attribute = attribute
        self.value = value
        self.data = data
        self.child = []
    def is_leaf(self):
        return self.data != None
class DecisionTreeClassifier:
    def __init__(self,max_depth=None,min_samples=2,feat_labels=None):
        """
        A decision tree classifier.
        Parameters
        ----------
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        min_samples : int, default=2
            Determines the mandatory minimum samples in an internal node to be considered for further splitting.
        feat_labels : list or array like. default=None
            Feature labels. Needed for better understanding of the tree.
        """
        self.predictor = feat_labels
        self.max_depth = max_depth
        self.min_samples = min_samples
    def _entropy(self,y):
        """
        calculate entropy of y
        Parameters
        ----------
        y: array like.
            contains the values to calculate entropy.
        Returns
        -------
        entropy: scaler, float.
        """
        n_unq = np.unique(y,return_counts=True)
        total = float(np.sum(n_unq[1]))
        sum = 0
        for i in range(len(n_unq[0])):
            probability = n_unq[1][i] / total
            sum -= (probability * np.log2(probability))
        return sum
    def _entropy_of_two(self,x,y):
        """
        Entropy of `y` splitted by values in `x`.
        Parameters
        ---------
        x: array like.
        y: array like
        """
        x_unq = np.unique(x)
        entropy = 0.0
        n = float(len(y))
        for i in range(len(x_unq)):
            idxs = np.argwhere(x==x_unq[i]).flatten()
            e = self._entropy(y[idxs])
            count = len(idxs)
            entropy += (e * (count/n))
        return entropy
    def _information_gain(self,x,y):
        before = self._entropy(y)
        after = self._entropy_of_two(x,y)
        return before - after
    def _best_split(self,X,y):
        """
        Returns the index of the attribute that returns largest information gain.
        Parameters
        ----------
        X: 2D array like, dataframe.
        y: array like
        Returns
        -------
        index: scalar. 
        """
        gain = -1
        idx = -1
        for i in range(X.shape[1]): 
            x = X[:,i]
            temp = self._information_gain(x,y)
            if temp > gain:
                gain = temp
                idx = i
        return idx,gain
    def _tree(self,X,y,node):
        """
        Generates the tree with recursive splits.
        Parameters
        ----------
        X: 2D array like, ndarray.
        y: array like
        node: Node object. 
        Returns
        -------
        node: Node object. 
        """
        col_id,gain = self._best_split(X,y)
        if node.depth>10 or y.shape[0] <=1 or gain<=0.01:
            values,counts = np.unique(y,return_counts=True)
            idx = np.argmax(counts)
            node.data = values[idx]
            return node
        uniques = np.unique(X[:,col_id])
        temp_childs = []
        for i in range(len(uniques)):
            idxs = np.argwhere(X[:,col_id]==uniques[i]).flatten()
            tmp_node = Node(depth=node.depth+1,attribute=col_id,value=uniques[i])
            temp_childs.append(self._tree(X[idxs,:],y[idxs],tmp_node))
        node.child = temp_childs
        return node
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            X = X.values
        if isinstance(y,pd.Series):
            y = y.values
        self.root = self._tree(X=X,y=y,node=Node(depth=0))
    def _traverse(self,node):
        if node.depth == 0:
            print("ROOT")
        else:
            if node.is_leaf():
                print("| "*node.depth,node.attribute,node.value,node.data, "*")
            else:
                print("| "*node.depth,node.attribute,node.value,node.data)
        for i in node.child:
            self._traverse(i)
    def _climb(self,x,node):
        for i in node.child:
            if x[i.attribute] == i.value: 
                if i.data!=None: 
                    return i.data
                else: 
                    temp = self._climb(x,i)
        return temp 
    def predict(self,X):
        if isinstance(X,pd.DataFrame):
            X = X.values
        results = []
        for i in range(X.shape[0]):
            x = X[i,:]
            result = self._climb(x,self.root)
            results.append(result)
        return results
