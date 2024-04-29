import pickle

import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

menu = [{"name": "KNN", "url": "p_knn"},
        {"name": "Линейная", "url": "p_linear"},
        {"name": "Логистическая", "url": "p_logistic"},
        {"name": "Дерево", "url": "p_three"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, по машинному обучению", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)


@app.route("/p_linear")
def f_linear():
    return render_template('lab2.html', title="Линейная регрессия", menu=menu)


@app.route("/p_logistic")
def f_logistic():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_three")
def f_three():
    return render_template('three.html', title="Дерево решений", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
