"""
THIS CODE IS MY OWN WORK. IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
Nathan Yang
"""


import argparse
import numpy as np
import pandas as pd
import math


class Knn(object):
    k = 0    # number of neighbors to use
    train_mat = None
    train_lab = None

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

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
        self.train_mat = xFeat.to_numpy()
        self.train_lab = y

        return self

    def dist(self, x1, x2):
        """
        Returns euclidean distance between x1 and x2, arrays of equal length that represent the features of two data points
        """
        sum_sqr_dif = 0
        for i in range(len(x1)):
            sum_sqr_dif += (x1[i]-x2[i])**2
        return math.sqrt(sum_sqr_dif)

    def classify(self, x):
        """
        Classify x using the labels, train_lab, of k-nearest neighbors in train_mat
        runtime: dist performs a calculation for every feature so d calculations. Then have to do that n times to get
        euclidean distance from x to every training point. From this we have runtime of O(nd). Then to get the k closest from
        those n distances takes O(kn). Overall runtime is then O(n(d+k)).
        """
        #distances from x to every entry in training data
        dists = np.array([self.dist(x, y) for y in self.train_mat])  
        #sorts the indices of dists based off the value at that index and shortens that list to include just the k closest   
        indices = np.argsort(dists)[:self.k]         
        #labels of the k nearest to x     
        k_nearest = [self.train_lab[i] for i in indices]
        #need to change the return to work for nonbinary labels perhaps mode function on k_nearest, 
        #but works for this assignment 
        if sum(k_nearest)/len(k_nearest) >= 0.5:
            return 1
        else: 
            return 0

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
        feat = np.array(xFeat)
        yHat = [self.classify(x) for x in feat] # variable to store the estimated class label
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    for i in range(len(yHat)):
        if yHat[i]-yTrue[i] == 0:
            acc += 1
    return acc/len(yHat)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
