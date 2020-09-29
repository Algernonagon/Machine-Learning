"""
THIS CODE IS MY OWN WORK. IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
Nathan Yang
"""

#I collaborated with the following classmates for this homework: Eric Gu, Harry Feng

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time

class Node(object):

    def __init__(self, matrix, array, split=None, left=None, right=None):
        self.matrix = matrix
        self.array = array
        self.split = split
        self.left = left
        self.right = right

def entropy(array):
    if len(array) == 0: return 0
    p1 = sum(array)/len(array)
    p0 = 1-p1
    if p1 == 0 or p0 == 0: return 0
    return 0-(p1*math.log2(p1)+p0*math.log2(p0))

def gini(array):
    if len(array) == 0: return 0
    p1 = sum(array)/len(array)
    p0 = 1-p1
    return 1-(p1**2+p0**2)

def split(matrix, array, feat, _split):
    #matrix is feature matrix and array is corresponding classifiers
    #feat is feature to split by and _split is the value of that feature to split by
    #matrix1 and array1 are the group of rows and classifiers that have a feat value >= _split
    matrix1 = matrix[ matrix[:, feat] >= _split ]
    array1 = array[ matrix[:, feat] >= _split ]
    #matrix1 and array1 are the group of rows and classifiers that have a feat value < _split
    matrix2 = matrix[ matrix[:, feat] < _split ]
    array2 = array[ matrix[:, feat] < _split ]
    return [matrix1, array1], [matrix2, array2]

def majority(array):
    if sum(array)/len(array) >= 0.5: return 1
    return 0

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        if criterion == "entropy": self.criterionFunc = entropy
        elif criterion == "gini": self.criterionFunc = gini
        else: print("[ERROR] Invalid criterion, options are: 'entropy' or 'gini'")

    def bestSplit(self, matrix, array):
        best_feat = 0
        best_value = matrix[0][0]
        best_split1, best_split2 = split(matrix, array, best_feat, best_value)
        best_split_val = len(best_split1[1])/len(array)*self.criterionFunc(best_split1[1]) + len(best_split2[1])/len(array)*self.criterionFunc(best_split2[1])
        for feat in range(len(matrix[0])):
            for value in matrix[:, feat]:
                split1, split2 = split(matrix, array, feat, value)
                if len(split1[1]) >= self.minLeafSample and len(split2[1]) >= self.minLeafSample:
                    criterion_val = len(split1[1])/len(array)*self.criterionFunc(split1[1]) + len(split2[1])/len(array)*self.criterionFunc(split2[1])
                    if criterion_val < best_split_val:
                        best_split1 = split1
                        best_split2 = split2
                        best_feat = feat
                        best_value = value
                        best_split_val = criterion_val
        return best_split1, best_split2, (best_feat, best_value)

    def trainRecur(self, node, depth):
        if depth >= self.maxDepth: return node, depth
        elif sum(node.array) == len(node.array) or sum(node.array) == 0: return node, depth
        elif len(node.matrix[0]) < 1: return node, depth
        else:
            left_split, right_split, split = self.bestSplit(node.matrix, node.array)
            if len(left_split[1]) < self.minLeafSample or len(right_split[1]) < self.minLeafSample: return node, depth
            else:
                #left_split[0] = np.delete(left_split[0], split[0], 1)
                #right_split[0] = np.delete(right_split[0], split[0], 1)
                depth += 1
                node.split = split
                node.left = Node(left_split[0], left_split[1])
                node.right = Node(right_split[0], right_split[1])
                self.trainRecur(node.left, depth)
                self.trainRecur(node.right, depth)
        
    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        # TODO do whatever you need
        matrix = xFeat.to_numpy()
        array = y.to_numpy()

        self.root = Node(matrix, array)
        self.trainRecur(self.root, 0)

        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        matrix = xFeat.to_numpy()
        for row in matrix:
            node = self.root
            while node.split != None:
                feat, val = node.split
                if row[feat] >= val: 
                    node = node.left
                else:
                    node = node.right
                #row = np.delete(row, feat, 0)
            yHat.append(majority(node.array))
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    start = time.time()
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    end = time.time()
    print("Time taken: ", end-start)


if __name__ == "__main__":
    main()
