import pandas as pd 
import numpy as np
class Node:
    def __init__(self,X=None,y=None,child=[],attribute=None,feature=None,value=None):
        self.X = X # X in this node
        self.y = y # y in this node
        self.depth = 1 # depth of this node
        self.child = child # if has child
        self.attribute = attribute # next split on attribute
        self.feature = feature # this branch threshold for parent's attribute
        self.value = value # if leaf
    def is_leaf(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self,min_leaf = 3, min_depth = None,feat_labels=None,target_label=None):
        """
        Decision Tree Regressor. Requires categorical features.
        """
        self.predictors = feat_labels
        self.target = target_label
        self.min_leaf = min_leaf
        self.min_depth = min_depth
        
    def standard_deviation(self,col1,col2=None,n_attr=1):
        """
        if col2 is given, then SD of col2 is calculated after dividing by 
        """
        if n_attr==1:
            length = len(col1)
            summation = np.sum(col1)
            average = summation/ float(length)
            standard_deviation = np.sqrt(np.sum([(value-average)**2 for value in col1]) / float(length))
        elif n_attr==2:
            # dividing values
#             print("col1",col1,"col2",col2)
            uniques = np.unique(col1) 
#             print("UNIQUES, ",col1.shape,col2.shape)
            standard_deviations = []
            counts = []
            if len(uniques)>1:
                for unq in uniques:
                    idx = np.argwhere(col1==unq).flatten()
    #                 print(col2,idx)
                    y = col2[idx]
                    sd = self.standard_deviation(y)
                    counts.append(len(idx))
                    standard_deviations.append(sd)
                standard_deviations = np.array(standard_deviations)
                probabilites = np.array([i/float(np.sum(counts)) for i in counts])
                standard_deviation = np.sum(standard_deviations*probabilites)
            else:
                standard_deviation = self.standard_deviation(col2)
        return standard_deviation
        
    def sdr(self,X,y):
        """
        Calculates standard deviation reduction of y given X
        """
#         print('IN SDR,',X.shape,y.shape)
        whole_sd  = self.standard_deviation(y)
        sdr = whole_sd - self.standard_deviation(X,y,n_attr=2)
        return sdr
    
    def best_split(self,X,y):
        """
        Returns the attribute idx for best split.
        If attribute is categorical, no further evaluation is needed
        """
        sd_reductions = []
#         print('In best split', X.shape,y.shape)
        for col in range(X.shape[1]):
            temp_X = X[:,col]
#             print('In best split after slice', temp_X.shape,y.shape)
            sd_reductions.append(self.sdr(temp_X,y))
        # has the attribute index for best spilit at that node
        best_idx = np.argmax(sd_reductions)
        return best_idx
    
    def grow_tree(self,node,depth=0):
        """
        Grows decisiont tree. Each call of this function modifies the current nodes.
        Require: 
        X = df, ndarray. 
        y = target series,array,list. 
        depth = depth of the tree.
        Ensure: 
        root = Node. Contains all child tree in itself.
        """
        # if node has only 1 item then make it leaf
        
        
        if node.X.shape[0] <=1:
            node.value = np.mean(node.y)
            print("made a leaf",node.feature,node.value)
            return node
        node.attribute = self.best_split(node.X,node.y) # main task of this func
        node.depth = depth
        splitters = np.unique(node.X[:,node.attribute])
        print(len(splitters))
        templist = []
        for i in splitters:
            indices = np.argwhere(node.X[:,node.attribute]==i).flatten()
            branch = Node(X=node.X[indices,:],y=node.y[indices],feature=i)
#             print('>>NODE TO BE SPLITTED',branch.X.shape, branch.y.shape)
            templist.append(self.grow_tree(branch,depth+1))
#             print("Child appended in depth",node.depth)
        node.child = templist
        return node
    
    def climb(self,node):
        if node.is_leaf():
            print("    "*node.depth,node.value,node.feature,"<<<< LEAF")
            return
        for i in node.child:
            print("    "*node.depth,self.predictors[node.attribute],"<<<< node")
            self.climb(i)
            
            
    def _follow(self,node,X):
        if node.is_leaf():
            print("gotcha!")
            return node.value
        split_col = node.attribute
        for i in node.child:
            print(i.feature,X[split_col])
            if i.feature == X[split_col]:
                print("   traversing")
                val = self._follow(i,X)
                return val
                
    def predict(self,X):
        """
        Predict y for a given X.
        Require: X = dataframe or ndarray of shape (samples,attributes)
        """
        if isinstance(X,pd.DataFrame):
            X = X.values
        n_samples = X.shape[0]
        n_attribute = X.shape[1]
        
        result = []
        for i in range(n_samples):
            data = X[i,:]
            print("Predicting ",data,data.shape)
            result.append(self._follow(self.root,data))
            
        return np.array(result)
    
    def fit(self,X,y):
        if not isinstance(X,np.ndarray):
            self.predictors = X.columns
            X = X.values
        if not isinstance(y,np.ndarray):
            y = y.values.flatten()
        
        parent = Node(X=X,y=y)
        self.root = self.grow_tree(parent)
        print("Tree has grown")
        print(len(self.root.child))
        self.climb(self.root)
