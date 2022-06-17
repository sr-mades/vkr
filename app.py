import flask
from flask import Flask, render_template, request
#import deeplearn as dl
import pandas as pd
import pickle
import numpy as np
import dataframe as df
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model

app = Flask(__name__, template_folder='templates')

#xdf = df.drop(columns='Соотношение матрица-наполнитель')
#ydf = df['Соотношение матрица-наполнитель']
    # разделение данных на обучающие и тестовые
#X_train, X_test, y_train, y_test = train_test_split(xdf.values, ydf.values, test_size=0.3, shuffle=True)

@app.route('/', methods=['post', 'get'])
def index():
    data = np.zeros((1,12))
    print(data)
    if request.method == 'GET':
        return render_template('index.html', result=3)
    if request.method == 'POST':

       # model_load = pickle.load(open('/Volumes/SRV/project/vkr/deeplearn.pkl', 'rb'))

        model_load = load_model('/Volumes/SRV/project/vkr/deeplearn.hd5')

        for i in range(12):
            data[0][i] = float(request.form['{0}'.format(i+2)])

        print(data)
        pred = model_load.predict(data)
        print(pred[0][0])
        return render_template('index.html', result=pred[0][0])


#app.add_url_rule('/', 'index', index)

if __name__ == '__main__':
    app.run()
