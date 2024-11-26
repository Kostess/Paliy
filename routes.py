import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from flask import render_template, Blueprint, request, redirect, url_for, current_app
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

from data import menu, metrics_classification_data, metrics_linear_data, food_names, pc_names
from my_models import loaded_model_knn, loaded_model_linear, loaded_model_logistic, loaded_model_tree, new_neuron, \
    model_fashion, model_food, model_pc, model_detection

scaler = StandardScaler()

routes_app = Blueprint('routes', __name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@routes_app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, по машинному обучению", menu=menu)


# Остальные маршруты остаются без изменений

@routes_app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='',
                               metrics=metrics_classification_data)
    if request.method == 'POST':
        try:
            x_new = np.array([[float(request.form['list1']), float(request.form['list2']), float(request.form['list3']),
                               float(request.form['list4'])]])
            pred = loaded_model_knn.predict(x_new)
            return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                                   class_model="Это: " + pred[0], metrics=metrics_classification_data)
        except Exception as e:
            return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                                   class_model="Ошибка: " + str(e), metrics=metrics_classification_data)


# Остальные маршруты остаются без изменений

@routes_app.route("/p_food", methods=['POST', 'GET'])
def f_food():
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
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = tf.keras.utils.load_img(file_path, target_size=(180, 180))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = model_food.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                predicted_class = food_names[np.argmax(score)]
                confidence = 100 * np.max(score)
                return render_template('neuro_food.html', title="Распознавание еды", menu=menu,
                                       class_model=f"Это: {predicted_class} с вероятностью {confidence:.2f}%")
            except Exception as e:
                return render_template('neuro_food.html', title="Распознавание еды", menu=menu,
                                       class_model="Ошибка: " + str(e))

    return redirect(url_for('routes.p_food'))


@routes_app.route("/p_pc", methods=['POST', 'GET'])
def f_pc():
    if request.method == 'GET':
        return render_template('pc.html', title="Распознавание комплектующих ПК", menu=menu)

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = tf.keras.utils.load_img(file_path, target_size=(160, 160))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = model_pc.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                predicted_class = pc_names[np.argmax(score)]
                confidence = 100 * np.max(score)
                return render_template('pc.html', title="Распознавание комплектующих ПК", menu=menu,
                                       class_model=f"Это: {predicted_class} с вероятностью {confidence:.2f}%")
            except Exception as e:
                return render_template('pc.html', title="Распознавание комплектующих ПК", menu=menu,
                                       class_model="Ошибка: " + str(e))

    return redirect(url_for('routes.p_pc'))


@routes_app.route("/p_detection", methods=['POST', 'GET'])
def p_detection():
    if request.method == 'GET':
        return render_template('detection.html', title="Детекция объектов", menu=menu)

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = model_detection(img)
                detections = results.pandas().xyxy[0]

                # Наложение рамок и меток на изображение
                for _, row in detections.iterrows():
                    label = f"{row['name']} {row['confidence']:.2f}"
                    cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
                    cv2.putText(img, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Сохранение обработанного изображения
                processed_filename = f"processed_{filename}"
                processed_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Передача относительного пути к изображению в шаблон
                processed_image_url = url_for('static', filename=f'uploads/{processed_filename}')
                print(processed_image_url)
                return render_template('detection.html', title="Детекция объектов", menu=menu, class_model="",
                                       processed_image_url=processed_image_url, error=None)
            except Exception as e:
                return render_template('detection.html', title="Детекция объектов", menu=menu, class_model="", detections=None,
                                       error="Ошибка: " + str(e))

    return redirect(url_for('p_detection'))


@routes_app.route('/doc_api')
def doc_api():
    return render_template('documentAPI.html', title="Документация по API", menu=menu)
