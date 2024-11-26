from flask import Blueprint, request, jsonify
from my_models import loaded_model_knn, loaded_model_linear, model_reg
import numpy as np

api_app = Blueprint('api', __name__)

@api_app.route('/api_sort', methods=['GET'])
def get_sort():
    request_data = request.get_json()
    x_new = np.array([[float(request_data['sepal_length']), float(request_data['sepal_width']), float(request_data['petal_length']), float(request_data['petal_width'])]])
    pred = loaded_model_knn.predict(x_new)
    return jsonify(sort=pred[0])

@api_app.route('/api_linear', methods=['GET'])
def get_linear():
    request_data = request.get_json()
    x_new = np.array([[float(request_data['height']), float(request_data['weight']), float(request_data['gender'])]])
    pred = loaded_model_linear.predict(x_new)
    return jsonify(result=pred[0])

@api_app.route('/api_reg', methods=['get'])
def predict_regression():
    input_data = np.array([[float(request.args.get('number_square')), float(request.args.get('number_rooms')), float(request.args.get('number_floors'))]])
    predictions = model_reg.predict(input_data)
    return jsonify(price=str(predictions[0][0]))