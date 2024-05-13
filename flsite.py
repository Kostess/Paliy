import pickle

import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

menu = [{"name": "Главная", "url": "/"},
        {"name": "KNN", "url": "p_knn"},
        {"name": "Линейная", "url": "p_linear"},
        {"name": "Логистическая", "url": "p_logistic"},
        {"name": "Дерево", "url": "p_three"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
loaded_model_linear = pickle.load(open('model/linearModel', 'rb'))
loaded_model_logistic = pickle.load(open('model/logistic_model', 'rb'))
loaded_model_tree = pickle.load((open('model/Tree_model', 'rb')))

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, по машинному обучению", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(x_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)


@app.route("/p_linear", methods=['POST', 'GET'])
def f_linear():
    if request.method == 'GET':
        return render_template('lab2.html', title="Линейная регрессия", menu=menu)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),]])
        pred = loaded_model_linear.predict(x_new)
        return render_template('lab2.html', title="Линейная регрессия", menu=menu, class_model=f"Ваш размер обуви: {pred}")


@app.route("/p_logistic", methods=['POST', 'GET'])
def f_logistic():
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']),]])
        pred = loaded_model_logistic.predict(x_new)
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model=f"Это: {pred}")

@app.route("/p_three", methods=['POST', 'GET'])
def f_three():
    if request.method == 'GET':
        return render_template('three.html', title="Дерево решений", menu=menu)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']), ]])
        pred = loaded_model_tree.predict(x_new)
        return render_template('three.html', title="Дерево решений", menu=menu, class_model=f"Это: {pred}")

if __name__ == "__main__":
    app.run(debug=True)
