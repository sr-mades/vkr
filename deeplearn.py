import keras.losses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_visualizer import visualizer
from keras.models import load_model

import pickle

def tensorf(df, column):
    # подготовка данных к обучению
    xdf = df.drop(columns=column)
    ydf = df[column]
    # разделение данных на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(xdf.values, ydf.values, test_size=0.3, shuffle=True)

    # нейронная сеть
    #n_X = xdf.shape[1] # размерность Х
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    plot_model(model, to_file='/Volumes/SRV/project/vkr/deeplearn.png', show_shapes=True, show_layer_names=True)
    #visualizer(model, format='png', view=True)
    # компилируем
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # обучаем
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        verbose=None)

    # валидация
    pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, pred)
    mean2 = metrics.mean_squared_error(y_test, pred)
    # Сохраниение модели
    pickle.dump(model, open('/Volumes/SRV/project/vkr/deeplearn.pkl', 'wb'))
    model.save('/Volumes/SRV/project/vkr/deeplearn.hd5')

    # model_load = pickle.load(open('/Volumes/SRV/project/vkr/deeplearn.pkl', 'rb'))
    # ml = model_load.predict(X_test)

    # model_load1 = load_model('/Volumes/SRV/project/vkr/deeplearn.hd5')
    # ml1 = model_load1.predict(X_test)

    # визуализация
    plt.title('обучение')
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()


    print(model.summary())
    print(X_test[:5])

    return