"""
THIS CODE IS MY OWN WORK. IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
Nathan Yang
"""

#I collaborated with the following classmates for this homework: Eric Gu, Harry Feng

import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def bestParams(estimator, params, k, xFeat, y):
    clf = GridSearchCV(estimator, params, cv=k)
    clf.fit(xFeat, y['label'])
    return clf.best_params_

def accAUC(model, xTrain, yTrain, xTest, yTest, rem=0):
    y = yTrain.drop(yTrain.sample(frac=rem).index)
    xFeat = xTrain.iloc[y.index]
    model.fit(xFeat, y['label'])
    yHat = model.predict(xTest)
    acc = metrics.accuracy_score(yTest['label'], yHat)
    yHat_prob = model.predict_proba(xTest)
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'], yHat_prob[:, 1])
    AUC = metrics.auc(fpr, tpr)
    return acc, AUC

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("folds",
                        type=int,
                        help="number of folds to use in cross validation for hyperparameter tuning")
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
    k = args.folds

    knParam = {"n_neighbors":range(1,50,2)}
    dtParam = {"criterion":["gini","entropy"], "max_depth":range(1,10), "min_samples_leaf":range(10,40)}
    knParam = bestParams(KNeighborsClassifier(), knParam, 10, xTrain, yTrain)
    dtParam = bestParams(DecisionTreeClassifier(), dtParam, 10, xTrain, yTrain)
    kn_clf = KNeighborsClassifier(knParam['n_neighbors'])
    dt_clf = DecisionTreeClassifier(criterion=dtParam['criterion'], max_depth=dtParam['max_depth'], min_samples_leaf=dtParam['min_samples_leaf'])

    kn_acc = []
    kn_AUC = []
    dt_acc = []
    dt_AUC = []
    for r in [0, 0.05, 0.1, 0.2]:
	    acc, AUC = accAUC(kn_clf, xTrain, yTrain, xTest, yTest, r)
	    kn_acc.append(acc)
	    kn_AUC.append(AUC)
	    acc, AUC = accAUC(dt_clf, xTrain, yTrain, xTest, yTest, r)
	    dt_acc.append(acc)
	    dt_AUC.append(AUC)

    df = pd.DataFrame([kn_acc, kn_AUC, dt_acc, dt_AUC], index=['kn accuracy', 'kn AUC', 'dt accuracy', 'dt AUC'], columns=['0% removed', '5% removed', '10% removed', '20% removed'])
    print(df)


if __name__ == "__main__":
    main()