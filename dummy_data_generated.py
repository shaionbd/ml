import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import requests
import json
from pandas.io.json import json_normalize

matplotlib.use('agg')


def generate_dataframe(min_val=0, max_val=100000, size=10000):
    total_cash_out_times = [np.random.randint(low=0, high=50) for _ in range(size)]
    total_cash_in_times = [np.random.randint(low=0, high=50) for _ in range(size)]
    cash_out_amount = [sum([np.random.randint(low=50, high=2000) for _ in range(total_cash_out_times[index])]) for index in range(size)]
    cash_in_amount = [sum([np.random.randint(low=50, high=2000) for _ in range(total_cash_in_times[index])]) for index in range(size)]

    data = {
        'cashInLimit': [20000 for _ in range(size)],
        'totalCashOutTimes': total_cash_out_times,
        'totalCashInTimes': total_cash_in_times,
        'cashOutAmount': cash_out_amount,
        'cashInAmount': cash_in_amount,
        'amountRemain': [(20000+cash_in_amount[index])-cash_out_amount[index] for index in range(size)],
        'date': datetime.now().date(),
        'time': [(datetime.now() - timedelta(minutes=index)).strftime("%H:%M") for index in range(size)]
    }
    return pd.DataFrame(data=data)


def generate_pkl(classifier):
    joblib.dump(classifier, 'bi.pkl')


def predict_from_pkl(predict_val):
    clf = joblib.load('bi.pkl')
    prediction = clf.predict(predict_val)
    predict = clf.predict_proba(predict_val) * 100
    return prediction, predict


def linear_regression_classifier(X_train, y_train, X_test, y_test, predict_val):
    print('Linear Regression')
    print('__________________________________________________')
    start = time.time()
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    end = time.time()
    print('Accuracy: ' + str(round(clf.score(X_test, y_test) * 100, 2)) + '%\n')
    print('Time: ' + str(round(end - start, 2)) + 'sec\n')
    print('__________________________________________________')

    prediction = clf.predict(predict_val)
    print(prediction)


def random_forest_classifier(X_train, y_train, X_test, y_test, predict_val):
    print('Random Forest')
    print('__________________________________________________')
    start = time.time()
    clf = RandomForestClassifier()
    clf = RFE(clf, n_features_to_select=100)
    clf = RFE(clf, 100)  # second parameter is the number of features
    clf.fit(X_train, y_train)
    end = time.time()
    print('Accuracy: ' + str(round(clf.score(X_test, y_test) * 100, 2)) + '%\n')
    print('Time: ' + str(round(end - start, 2)) + 'sec\n')
    print('Num of Features: ' + str(clf.n_features_) + '\n')
    # print('Features sorted by their rank:' + '\n')
    # print(str(sorted(zip(map(lambda x: round(x, 4), clf.ranking_), features))))
    print('__________________________________________________')

    # generate pkl file for future use without fitting
    generate_pkl(clf)

    prediction = clf.predict(predict_val)
    predict = clf.predict_proba(predict_val) * 100

    for i in range(len(predict)):
        print("For ", predict_val[i])
        print("Need ", str(int(predict[i][1])) + "%")
        print("No Need ", str(int(predict[i][0])) + "%")
        print("----------------------------")

    print(prediction)


def decision_tree_classifier(X_train, y_train, X_test, y_test, predict_val):
    print('Decision Tree')
    print('__________________________________________________')
    start = time.time()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    end = time.time()
    print('Accuracy: ' + str(round(clf.score(X_test, y_test) * 100, 2)) + '%\n')
    print('Time: ' + str(round(end - start, 2)) + 'sec\n')
    print('Num of Features: ' + str(clf.n_features_) + '\n')
    print('__________________________________________________')

    prediction = clf.predict(predict_val)
    predict = clf.predict_proba(predict_val) * 100

    for i in range(len(predict)):
        print("For ", predict_val[i])
        print("Need ", str(int(predict[i][1])) + "%")
        print("No Need ", str(int(predict[i][0])) + "%")
        print("----------------------------")

    print(prediction)


def k_neighbors_classifier(X_train, y_train, X_test, y_test, predict_val):
    print('K Neighbors')
    print('__________________________________________________')
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(X_train, y_train)
    end = time.time()
    print('Accuracy: ' + str(round(clf.score(X_test, y_test) * 100, 2)) + '%\n')
    print('Time: ' + str(round(end - start, 2)) + 'sec\n')
    print('__________________________________________________')

    prediction = clf.predict(predict_val)
    predict = clf.predict_proba(predict_val) * 100

    for i in range(len(predict)):
        print("For ", predict_val[i])
        print("Need ", str(int(predict[i][1])) + "%")
        print("No Need ", str(int(predict[i][0])) + "%")
        print("----------------------------")

    print(prediction)


def predict_for_api_data():
    data = [
        {'id': 23, 'cashInLimit': 20000, 'totalCashOutTimes': 1, 'totalCashInTimes': 4, 'cashOutAmount': 8000,
         'cashInAmount': 10000, 'amountRemain': 4000},
        {'id': 24, 'cashInLimit': 20000, 'totalCashOutTimes': 2, 'totalCashInTimes': 2, 'cashOutAmount': 3000,
         'cashInAmount': 6000, 'amountRemain': 3000},
        {'id': 25, 'cashInLimit': 20000, 'totalCashOutTimes': 38, 'totalCashInTimes': 28, 'cashOutAmount': 80000,
         'cashInAmount': 73000, 'amountRemain': 7000},
        {'id': 26, 'cashInLimit': 20000, 'totalCashOutTimes': 22, 'totalCashInTimes': 19, 'cashOutAmount': 55000,
         'cashInAmount': 45000, 'amountRemain': 6000},
        {'id': 27, 'cashInLimit': 20000, 'totalCashOutTimes': 3, 'totalCashInTimes': 5, 'cashOutAmount': 12000,
         'cashInAmount': 9000, 'amountRemain': 3000},
        {'id': 28, 'cashInLimit': 20000, 'totalCashOutTimes': 54, 'totalCashInTimes': 45, 'cashOutAmount': 74000,
         'cashInAmount': 47000, 'amountRemain': 27000},
        {'id': 29, 'cashInLimit': 20000, 'totalCashOutTimes': 17, 'totalCashInTimes': 19, 'cashOutAmount': 20000,
         'cashInAmount': 19000, 'amountRemain': 5000},
        {'id': 30, 'cashInLimit': 20000, 'totalCashOutTimes': 19, 'totalCashInTimes': 11, 'cashOutAmount': 12000,
         'cashInAmount': 7500, 'amountRemain': 4500}
    ]
    df = json_normalize(data)
    features = ['cashInLimit', 'totalCashOutTimes', 'totalCashInTimes', 'cashOutAmount',
                'cashInAmount', 'amountRemain']

    df_predict = df[features]

    print(df_predict.as_matrix())

    # # predict from recommended system
    prediction, predict = predict_from_pkl(df_predict)
    df['need'] = [p[1] for p in predict]
    df['noNeed'] = [p[0] for p in predict]
    df['prediction'] = prediction
    return df


if __name__ == '__main__':
    # print('__________________________________________________')
    # start = time.time()
    # # out put pattern
    # # agent_id, limit_amount, cash_in_times, cash_out_times, cash_in_amount, cash_out_amount,
    # # predict_need, predict_no_need, final_predict
    # df = predict_for_api_data()
    # print(df[['id', 'amountRemain', 'need', 'noNeed', 'prediction']])
    # end = time.time()
    # print('Time: ' + str(round(end - start, 2)) + 'sec\n')
    # print('__________________________________________________')

    limit = 20000
    # df = generate_dataframe(size=50000)
    # df.to_csv("bi.csv", index=False)
    df = pd.read_csv("bi_with_predict.csv")
    features = ['cashInLimit', 'totalCashOutTimes', 'totalCashInTimes', 'cashOutAmount',
                'cashInAmount', 'amountRemain']
    min_limit = int(limit - (limit*0.8))
    y = df['needAmount'].values
    df = df[features]
    X = df.values
    # df['needAmount'] = np.where(df['amountRemain'] <= 8000, 1, 0)
    # df.to_csv("bi_with_predict.csv", index=False)
    predict_val = [[20000, 1, 4, 8000, 10000, 4000], [20000, 2, 2, 3000, 6000, 3000], [20000, 38, 28, 80000, 73000, 7000]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

    random_forest_classifier(X_train, y_train, X_test, y_test, predict_val)
    # print("===========================================================================================")
    # linear_regression_classifier(X_train, y_train, X_test, y_test, predict_val)
    # print("===========================================================================================")
    # decision_tree_classifier(X_train, y_train, X_test, y_test, predict_val)
    # print("===========================================================================================")
    # k_neighbors_classifier(X_train, y_train, X_test, y_test, predict_val)
