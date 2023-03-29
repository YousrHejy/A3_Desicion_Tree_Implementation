# A3_Desicion_Tree_Implementation
 
#### This is an implementation of a basic decision tree algorithm, using the ID3 algorithm with information gain as the splitting criterion. Here is a brief overview of the different parts of the code:

1. Node class: This represents a node in the decision tree. Each node has an attribute (the feature used for splitting), a dictionary of branches (representing the possible values of the attribute and the corresponding child nodes), and a label (the predicted class if this is a leaf node).

2. DecisionTree class: This represents the decision tree itself. It has several methods for building and using the tree, including:

- Most_Common_Label: A helper method that returns the most common class label in a list of labels.

- Entropy: A method that calculates the entropy of a set of labels.

- InformationGain: A method that calculates the information gain of a feature based on its ability to separate the labels.

- Choose_Best_Feature: A method that selects the best feature to split on based on its information gain.

- _build_tree: A recursive method that builds the decision tree by splitting the data based on the best feature at each node.

- fit: A method that trains the decision tree on a dataset.

- Classify: A recursive method that classifies a new data point by traversing the decision tree.

- predict: A method that predicts the class labels for a set of data points.

Then Use this class to predict if the pateint has cardiovascular disease or not and compare the accuracies with the DT in sklearn.
