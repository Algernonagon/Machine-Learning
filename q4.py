"""
THIS CODE IS MY OWN WORK. IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
Nathan Yang
"""


import argparse
import numpy as np
import pandas as pd

import knn


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN
    xTrainStd, xTestStd = xTrain.copy(deep=True), xTest.copy(deep=True)
    stdTrain = xTrain.std(axis=0)
    avgTrain = xTrain.mean(axis=0)
    stdTest = xTest.std(axis=0)
    avgTest = xTest.mean(axis=0)
    for col in xTrain.columns:
        xTrainStd[col] = (np.array(xTrain[col])-avgTrain[col])/stdTrain[col]
        xTestStd[col] = (np.array(xTest[col])-avgTest[col])/stdTest[col]
    return xTrainStd, xTestStd 


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN
    xTrainMM, xTestMM = xTrain.copy(deep=True), xTest.copy(deep=True)
    minTrain = xTrain.min(axis=0)
    maxTrain = xTrain.max(axis=0)
    minTest = xTest.min(axis=0)
    maxTest = xTest.max(axis=0)
    for col in xTrain.columns:
        xTrainMM[col] = (np.array(xTrain[col])-minTrain[col])/(maxTrain[col]-minTrain[col])
        xTestMM[col] = (np.array(xTest[col])-minTest[col])/(maxTest[col]-minTest[col])
    return xTrainMM, xTestMM


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    # TODO FILL IN
    xTrainIrr, xTestIrr = xTrain.copy(deep=True), xTest.copy(deep=True)
    xTrainIrr['noise1'] = np.random.randn(len(xTrain.index))
    xTrainIrr['noise2'] = np.random.randn(len(xTrain.index))
    xTestIrr['noise1'] = np.random.randn(len(xTest.index))
    xTestIrr['noise2'] = np.random.randn(len(xTest.index))
    return xTrainIrr, xTestIrr 


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
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
    model = knn.Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn.accuracy(yHatTest, yTest['label'])
    

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

if __name__ == "__main__":
    main()
