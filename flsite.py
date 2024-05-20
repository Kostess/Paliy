import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

menu = [{"name": "Главная", "url": "/"},
        {"name": "KNN", "url": "p_knn"},
        {"name": "Линейная", "url": "p_linear"},
        {"name": "Логистическая", "url": "p_logistic"},
        {"name": "Дерево", "url": "p_three"},
        {"name": "Документация по API", "url": "doc_api"},]

metrics_linear_data = [
    {"name": "MSE", "value": "1.3509243001909827e-29"},
    {"name": "RMSE", "value": "6.754621500954914e-30"},
    {"name": "MSPE", "value": "5.403697200763931e-29"},
    {"name": "MAPE", "value": "8.500000000000002"},
    {"name": "SMAPE", "value": "1.831867990631502e-15"},
    {"name": "MASE", "value": "7.327471962526033e-15"},
    {"name": "MRE", "value": "inf"},
    {"name": "RMSLE", "value": "2.9624445261614147e-15"},
    {"name": "R-квадрат", "value": "1.0"},
    {"name": "Скорректированный R-квадрат", "value": "0.7986743400332134"},
    ]

metrics_classification_data = [
    {"name": "Confusion matrix", "type": "logistic", "value": {"Predicted": [73, 15], "Actual": [10, 82]}},
    {"name": "Confusion matrix", "type": "KNN", "value": {"Predicted": [72, 16], "Actual": [14, 78]}},
    {"name": "Confusion matrix", "type": "three", "value": {"Predicted": [70, 18], "Actual": [21, 71]}},
    {"name": "Accuracy", "type": "logistic", "value": 0.8611111111111112},
    {"name": "Accuracy", "type": "KNN", "value": 0.8333333333333334},
    {"name": "Accuracy", "type": "three", "value": 0.7833333333333333},
    {"name": "Precision", "type": "logistic", "value": 0.845360824742268},
    {"name": "Precision", "type": "KNN", "value": 0.8297872340425532},
    {"name": "Precision", "type": "three", "value": 0.797752808988764},
    {"name": "Recall", "type": "logistic", "value": 0.8913043478260869},
    {"name": "Recall", "type": "KNN", "value": 0.8478260869565217},
    {"name": "Recall", "type": "three", "value": 0.7717391304347826},
]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
loaded_model_linear = pickle.load(open('model/linearModel', 'rb'))
loaded_model_logistic = pickle.load(open('model/logistic_model', 'rb'))
loaded_model_tree = pickle.load((open('model/Tree_model', 'rb')))


@app.route("/")
def index():
    return render_template('index.html',
                           title="Лабораторные работы, по машинному обучению",
                           menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
    if request.method == 'GET':
        return render_template('lab1.html',
                               title="Метод k -ближайших соседей (KNN)",
                               menu=menu,
                               class_model='',
                               metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(x_new)
        return render_template('lab1.html',
                               title="Метод k -ближайших соседей (KNN)",
                               menu=menu,
                               class_model="Это: " + pred[0],
                               metrics=metrics_classification_data)


@app.route("/p_linear", methods=['POST', 'GET'])
def f_linear():
    if request.method == 'GET':
        return render_template('lab2.html',
                               title="Линейная регрессия",
                               menu=menu,
                               metrics=metrics_linear_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),]])
        pred = loaded_model_linear.predict(x_new)
        return render_template('lab2.html',
                               title="Линейная регрессия",
                               menu=menu,
                               class_model=f"Ваш размер обуви: {pred[0]}",
                               metrics=metrics_linear_data)


@app.route("/p_logistic", methods=['POST', 'GET'])
def f_logistic():
    if request.method == 'GET':
        return render_template('lab3.html',
                               title="Логистическая регрессия",
                               menu=menu,
                               metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']), ]])
        pred = loaded_model_logistic.predict(x_new)
        return render_template('lab3.html',
                               title="Логистическая регрессия",
                               menu=menu,
                               class_model=f"Это: {pred[0]}",
                               metrics=metrics_classification_data)


@app.route("/p_three", methods=['POST', 'GET'])
def f_three():
    if request.method == 'GET':
        return render_template('three.html',
                               title="Дерево решений",
                               menu=menu,
                               metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']), ]])
        pred = loaded_model_tree.predict(x_new)
        return render_template('three.html',
                               title="Дерево решений",
                               menu=menu,
                               class_model=f"Это: {pred[0]}",
                               metrics=metrics_classification_data)


@app.route('/api_sort', methods=['GET'])
def get_sort():
    request_data = request.get_json()
    x_new = np.array([[float(request_data['sepal_length']),
                       float(request_data['sepal_width']),
                       float(request_data['petal_length']),
                       float(request_data['petal_width'])]])
    pred = loaded_model_knn.predict(x_new)

    return jsonify(sort=pred[0])


@app.route('/api_linear', methods=['GET'])
def get_linear():
    request_data = request.get_json()
    x_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['gender'])]])
    pred = loaded_model_linear.predict(x_new)

    return jsonify(result=pred[0])


@app.route('/doc_api')
def doc_api():
    return render_template('documentAPI.html', title="Документация по API", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
