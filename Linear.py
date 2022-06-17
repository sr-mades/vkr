import numpy as np
import pandas as pd
import sklearn.metrics

import dataframe as d
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

def linearreg(df, column):
    # подготовка данных к обучению
    xdf = df.drop(columns=column)
    ydf = df[column]
    # разделение данных на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(xdf.values, ydf.values, test_size=0.3, shuffle=True)
    # обучение модели
    linear_regression = LinearRegression().fit(X=X_train, y=y_train)
    # тестирование модели
    predicted = linear_regression.predict(X_test)
    expected = y_test
    # метрики
    r2 = metrics.r2_score(expected, predicted)
    mean2 = metrics.mean_squared_error(expected, predicted)
    # выбор лучшей модели
    estimators = {'LinearRegression': linear_regression, 'ElasticNet': ElasticNet(), 'Lasso': Lasso(), 'Ridge': Ridge()}
    for estimator_name, estimator_object in estimators.items():
        kfold = KFold(n_splits=10, shuffle=True)
        scores = cross_val_score(estimator=estimator_object, X=xdf.values, y=ydf.values, cv=kfold)
        print('{0}: '.format(estimator_name) + '{0: .3f}'.format(scores.mean()))
    # for k in range(1, 20, 2):
    #     kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    #     knn = LinearRegression(n_jobs=k)
    #     scores = cross_val_score(estimator=knn, X=xdf.values, y=ydf.values, cv=kfold)
    #     print(f'k={k:<2}; mean accuracy={scores.mean():.2%}; ' +
    #           f'standard deviation={scores.std():.2%}')
    # вывод данных
    # indexies = pd.Series(d.df.columns)
    # reg = pd.concat([indexies, pd.Series(linear_regression.coef_)], axis=1, join='inner')
    # #reg['intercept', 'R2'] = pd.Series(linear_regression.intercept_, r2)
    pickle.dump(linear_regression, open('/Volumes/SRV/project/vkr/linearreg.pkl', 'wb'))

  #  model_load = pickle.load(open('/Volumes/SRV/project/vkr/linearreg.pkl', 'rb'))
   # ml = model_load.predict(X_test)

    print(mean2)
    print(predicted[:5], y_test[:5])
  #  print(linear_regression.coef_, linear_regression.intercept_)

    return linear_regression.coef_, linear_regression.intercept_