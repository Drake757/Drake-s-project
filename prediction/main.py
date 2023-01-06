import pickle

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pip._vendor.webencodings import labels
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR



def evaluate_predictions(predictions, true):

    mae = np.mean(abs(predictions - true))

    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    return mae, rmse

def evaluate(X_train, X_test, y_train, y_test):
    # 模型名称
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')



    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=50)

    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)

    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # 误差标准
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        # 将结果插入结果框
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]

        # 中值基准度量
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    return results




if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei')  # 解决Seaborn中文显示问题
    student = pd.read_csv('student-mat-pass-or-fail.csv')
    G3 = student[['G3']]
    failures = student[['failures']]
    list = G3.values.tolist()
    list2 = failures.values.tolist()
    G3list=[]
    failures_list=[]
    for i in range(0, 395):
        G3list.append(list[i][0])
    for i in range(0, 395):
        failures_list.append(list2[i][0])
    print(G3.describe())
    labels = student['G3']

    student = student.drop(['school'], axis='columns')
    student = pd.get_dummies(student)
    most_correlated = student.corr().abs()['G3'].sort_values(ascending=False)

    most_correlated = most_correlated[:6]

    print(most_correlated)

    X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size=0.25, random_state=42)
    X_test = X_test[['failures', 'Medu', 'higher', 'absences', 'Fedu','G1','G2','G3']]
    print(X_test)
    X_train = X_train[['failures','Medu','higher','absences','Fedu','G1','G2']]
    X_test = X_test[['failures','Medu','higher','absences','Fedu','G1','G2']]






    regr = LinearRegression()
    regr.fit(X_train, y_train)
    a=regr.predict(X_test)
    print(a)
    b=regr.score(X_test,y_test)
    print("degree of fitting")
    print(b)
    filename = 'LR_Model for more features and g1 g2'

    pickle.dump(regr, open(filename, 'wb'))


    # failures_swarmplot = sns.swarmplot(x=student['failures'], y=student['G3'])
    #
    # failures_swarmplot.axes.set_title('失败次数少的学生分数更高吗？', fontsize=30)
    #
    # failures_swarmplot.set_xlabel('失败次数', fontsize=20)
    #
    # failures_swarmplot.set_ylabel('最终成绩', fontsize=20)
    #
    # plt.show()

    # X_train, X_test, y_train, y_test = train_test_split(failures, labels, test_size = 0.25, random_state=42)
    #
    #
    # median_pred = X_train['G3'].median()
    #
    # median_preds = [median_pred  for i  in range(len(X_test))]
    #
    # true = X_test['G3']
    #
    # mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
    #
    # print('Median Baseline MAE: {:.4f}'.format(mb_mae))
    #
    # print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))
    #
    # print('X_train', X_train)
    # results = evaluate(X_train, X_test, y_train, y_test)
    # print(results)
    #
    # plt.figure(figsize=(12, 8))
    #
    # # 平均绝对误差
    # ax = plt.subplot(1, 2, 1)
    # results.sort_values('mae', ascending=True).plot.bar(y='mae', color='b', ax=ax, fontsize=20)
    # plt.title('平均绝对误差', fontsize=20)
    # plt.ylabel('MAE', fontsize=20)
    #
    # # 均方根误差
    # ax = plt.subplot(1, 2, 2)
    # results.sort_values('rmse', ascending=True).plot.bar(y='rmse', color='r', ax=ax, fontsize=20)
    # plt.title('均方根误差', fontsize=20)
    # plt.ylabel('RMSE', fontsize=20)
    # plt.tight_layout()
    # plt.show()
    # model = LinearRegression()
    #
    # model.fit(X_train, y_train)
    #
    # filename = 'LR_Model'
    #
    # pickle.dump(model, open(filename, 'wb'))
    #
    # f = open('LR_Model', 'rb')  # 注意此处model是rb
    # s = f.read()
    # model = pickle.loads(s)
    # X_train = X_train.drop('G3', axis='columns')
    # X_test = X_test.drop('G3', axis='columns')
    # print('X_train',X_train)
    # print(X_test)
    # print(y_train)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_train)
    # print(predictions)
    # print(predictions[40])


