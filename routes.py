import os

import numpy as np
from PIL import Image
from flask import render_template, Blueprint, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

from data import menu, metrics_classification_data, metrics_linear_data, food_names
from models import loaded_model_knn, loaded_model_linear, loaded_model_logistic, loaded_model_tree, new_neuron, \
    model_fashion, model_food

scaler = StandardScaler()

app = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, по машинному обучению", menu=menu)

@app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='', metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']), float(request.form['list4'])]])
        pred = loaded_model_knn.predict(x_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model="Это: " + pred[0], metrics=metrics_classification_data)

@app.route("/p_linear", methods=['POST', 'GET'])
def f_linear():
    if request.method == 'GET':
        return render_template('lab2.html', title="Линейная регрессия", menu=menu, metrics=metrics_linear_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']), ]])
        pred = loaded_model_linear.predict(x_new)
        return render_template('lab2.html', title="Линейная регрессия", menu=menu, class_model=f"Ваш размер обуви: {pred[0]}", metrics=metrics_linear_data)

@app.route("/p_logistic", methods=['POST', 'GET'])
def f_logistic():
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']), float(request.form['list4']), float(request.form['list5']), float(request.form['list6']), float(request.form['list7']), ]])
        pred = loaded_model_logistic.predict(x_new)
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, class_model=f"Это: {pred[0]}", metrics=metrics_classification_data)

@app.route("/p_three", methods=['POST', 'GET'])
def f_three():
    if request.method == 'GET':
        return render_template('three.html', title="Дерево решений", menu=menu, metrics=metrics_classification_data)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']), float(request.form['list4']), float(request.form['list5']), float(request.form['list6']), float(request.form['list7']), ]])
        pred = loaded_model_tree.predict(x_new)
        return render_template('three.html', title="Дерево решений", menu=menu, class_model=f"Это: {pred[0]}", metrics=metrics_classification_data)

@app.route("/p_neuro", methods=['GET', 'POST'])
def neuro():
    if request.method == 'GET':
        return render_template('neuro.html', title="Первый нейрон", menu=menu)
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']), ]])
        x_scaler = scaler.fit_transform(x_new)
        predictions = new_neuron.forward(x_scaler)
        return render_template('neuro.html', title="Первый нейрон", menu=menu, class_model="Это: " + str(*np.where(predictions >= 0.5, 'Собака', 'Кошка')))

@app.route("/p_fashion", methods=['POST', 'GET'])
def f_fashion():
    if request.method == 'GET':
        return render_template('neuro_fashion.html', title="Распознавание одежды", menu=menu)

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            img = Image.open(file_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            pred = model_fashion.predict(img_array)
            class_label = np.argmax(pred, axis=1)
            class_names = ['Футболка/топ', 'Брюки', 'Свитер', 'Платье', 'Пальто', 'Сандали', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']
            return render_template('neuro_fashion.html', title="Распознавание одежды", menu=menu, class_model=f"Это: {class_names[class_label[0]]}")
    return redirect(url_for('routes.p_fashion'))


@app.route("/p_food", methods=['POST', 'GET'])
def f_fashion():
    if request.method == 'GET':
        return render_template('neuro_food.html', title="Распознавание еды", menu=menu)

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            img = Image.open(file_path).convert('L')
            img = img.resize((180, 180))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 180, 180, 1)

            pred = model_food.predict(img_array)
            class_label = np.argmax(pred, axis=1)
            return render_template('neuro_food.html', title="Распознавание еды", menu=menu, class_model=f"Это: {food_names[class_label[0]]}")
    return redirect(url_for('routes.p_food'))

@app.route('/doc_api')
def doc_api():
    return render_template('documentAPI.html', title="Документация по API", menu=menu)
