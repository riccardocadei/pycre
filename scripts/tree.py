import numpy as np
import pandas as pd

class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, node_left=None, node_right=None, split_stat=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.node_left = node_left
        self.node_right = node_right
        self.split_stat = split_stat
        self.value = value

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, max_depth=3, node_size=10, criterion='het'):
        self.max_depth = max_depth
        self.node_size = node_size
        self.criterion = criterion
        self.root = None
    
    def __splitting_statistic(self, left_y, right_y):
        '''
        Helper function, calculates splitting statistics from the 2 child nodes.
        '''
        mean_left_y = np.mean(left_y)
        mean_right_y = np.mean(right_y)
        
        if self.criterion == 'mse':
            res2_left_y = (left_y - mean_left_y)**2
            res2_right_y = (right_y - mean_right_y)**2
            return -np.mean(np.concatenate((res2_left_y, res2_right_y)))

        elif self.criterion == 'het':
            var_left_y = np.var(left_y)
            var_right_y = np.var(right_y)
            if var_left_y+var_right_y == 0:
                return float('inf')
            else:
                return (mean_left_y-mean_right_y)**2 / (var_left_y+var_right_y)
        else:
            raise ValueError(f'Unknown criterion: {self.criterion}')
        
    def __best_split(self, X, y, used_features=[]):
        '''
        Helper function, calculates the best split for a given dataset
        '''
        best_split = {}
        best_split_stat = -float('inf')
        _, n_cols = X.shape
        
        f_idxs = list(set(range(n_cols)) - set(used_features))
        for f_idx in f_idxs:
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
                    split_stat = self.__splitting_statistic(y_left, y_right)
                    if split_stat > best_split_stat:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'split_stat': split_stat
                        }
                        best_split_stat = split_stat
        return best_split
    
    def __build(self, X, y, depth=0, used_features=[]):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        '''
        n_rows, n_cols = X.shape
        
        # Internal node
        if n_rows >= self.node_size and depth < min(self.max_depth,n_cols):
            best = self.__best_split(X, y, used_features)
            left = self.__build(
                X=best['df_left'][:, :-1], 
                y=best['df_left'][:, -1], 
                depth=depth + 1,
                used_features=used_features + [best['feature_index']]
            )
            right = self.__build(
                X=best['df_right'][:, :-1], 
                y=best['df_right'][:, -1], 
                depth=depth + 1,
                used_features=used_features + [best['feature_index']]
            )
            return Node(
                feature=self.X_columns[best['feature_index']], 
                threshold=best['threshold'], 
                node_left=left, 
                node_right=right, 
                split_stat=best['split_stat']
            )
        # Leaf node 
        return Node(value=np.mean(y))
    
    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: pd.dataframe, features
        :param y: np.array or list, target
        :return: None
        '''
        if isinstance(X, pd.DataFrame):
            self.X_columns = X.columns
            X = np.array(X)
            y = np.array(y)
        else:
            raise ValueError('X must be a pandas DataFrame')
        self.root = self.__build(X, y)

    def get_rules(self, decimal=2):
        '''
        Function used to extract the decision rules from a decision tree.
        '''
        rule = []
        rules = []
        def recurse(node, rule, rules):
            if node.feature != None:
                name = node.feature
                threshold = node.threshold
                rule_left = rule + [f"(X['{name}']<={threshold:.{decimal}f})"]
                rules += [rule_left]
                recurse(node.node_left, rule_left, rules)
                rule_right = rule + [f"(X['{name}']>{threshold:.{decimal}f})"]
                rules += [rule_right]
                recurse(node.node_right, rule_right, rules)
                
        recurse(self.root, rule, rules)
        for i, rule in enumerate(rules):
            rules[i] = ' & '.join(sorted(rule))
        return rules
        
    def __predict(self, x, node):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        '''
        # Leaf node
        if node.value != None:
            return node.value
        feature_value = x[node.feature]
        
        # Internal node
        if feature_value <= node.threshold:
            return self.__predict(x=x, node=node.node_left)
        if feature_value > node.threshold:
            return self.__predict(x=x, node=node.node_right)
        
    def predict(self, X):
        '''
        Function used to classify new instances.
        '''
        return X.apply(lambda x: self.__predict(x, self.root), axis=1)



