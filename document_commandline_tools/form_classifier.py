#!/usr/bin/env python3

import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def trainModels(src, dest):

    df = pd.read_csv(src)

    # print(df.dtypes)
    # print(df.head(10))
    # print(df.columns)

    columns_to_drop = []

    # drop categorical
    for field in df.columns:
        if df[field].dtype == 'object':
            columns_to_drop.append(field)

    df = df.drop(columns_to_drop, axis=1)
    labels = df['research_consent']
    df = df.drop(['research_consent'], axis=1)

    for column in df.columns:
        if df[column].isnull().values.any():
            df[column].fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=.4)

    # # correlations
    # corr_matrix = train_set.corr()
    # print(corr_matrix)
    # print()

    models = []

    result_rows = []

    models.append(("LogisticRegression",LogisticRegression(solver='lbfgs', max_iter=10000)))
    models.append(("SVC",SVC(gamma='scale', max_iter=-1)))
    models.append(("KNeighbors",KNeighborsClassifier()))
    models.append(("DecisionTree",DecisionTreeClassifier()))
    models.append(("RandomForest",RandomForestClassifier(n_estimators=100)))
    rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                    max_depth=10, random_state=0, max_features=None)
    models.append(("RandomForest2",rf2))
    models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))

    count = 1
    results_frame = pd.DataFrame()

    for name, model in models:
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        crossV_score = cross_val_score(model, X_test, y_test,  cv=7)

        results_frame.loc[count, 'Model'] = name
        results_frame.loc[count, 'Score'] = score
        results_frame.loc[count, 'Cross_Validation_score'] = crossV_score.mean()
        results_frame.loc[count, 'Cross_Validation_score.std'] = crossV_score.std() * 2

        count = count + 1

    print(results_frame)
    # results_frame.to_csv(dest)

if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--src", help=".csv file to train a model")
    parser.add_argument("--dest", help=".csv file to train a model")
    args = parser.parse_args()

    # run puppy, run
    trainModels(args.src, args.dest)
