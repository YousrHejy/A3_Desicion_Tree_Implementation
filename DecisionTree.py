#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from collections import Counter

class Node:
    def __init__(self, attribute=None, branches=None, label=None):
        self.attribute = attribute
        self.branches = branches if branches else {}
        self.label = label

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def Most_Common_Label(self, y):
        count_dict = Counter(y)
        return count_dict.most_common(1)[0][0]
    
    def Entropy(self, y):
        total_count = len(y)
        count_dict = Counter(y)
        entropy = 0.0
        for count in count_dict.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        return entropy
    
    def InformationGain(self, X, y, attribute):
        parent_entropy = self.Entropy(y)
        weighted_child_entropy = 0.0
        for value in set(X[:, attribute]):
            X_v = X[X[:, attribute] == value]
            y_v = y[X[:, attribute] == value]
            weighted_child_entropy += len(X_v) / len(X) * self.Entropy(y_v)
        return parent_entropy - weighted_child_entropy

    def Choose_Best_Feature(self, X, y):
        best_attribute = None
        best_info_gain = -math.inf
        for attribute in range(X.shape[1]):
            info_gain = self.InformationGain(X, y, attribute)
            if info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = info_gain
        return best_attribute
    
    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(X) < self.min_samples_split:
            return Node(label=self.Most_Common_Label(y))
        
        best_attribute = self.Choose_Best_Feature(X, y)
        node = Node(attribute=best_attribute)
        for value in set(X[:, best_attribute]):
            X_v = X[X[:, best_attribute] == value]
            y_v = y[X[:, best_attribute] == value]
            node.branches[value] = self._build_tree(X_v, y_v, depth + 1)
        return node
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def Classify(self, x, node):
        if node.label is not None:
            return node.label
        value = x[node.attribute]
        if value not in node.branches:
            return self.Most_Common_Label([child.label for child in node.branches.values()])
        return self.Classify(x, node.branches[value])
    
    def predict(self, X):
        prediction_list = []
        for x in X:
            prediction_list.append(self.Classify(x, self.root))
        return prediction_list


# In[ ]:




