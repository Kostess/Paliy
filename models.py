import pickle
import tensorflow as tf
from model.Neuro import SingleNeuron

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
loaded_model_linear = pickle.load(open('model/linearModel', 'rb'))
loaded_model_logistic = pickle.load(open('model/logistic_model', 'rb'))
loaded_model_tree = pickle.load((open('model/Tree_model', 'rb')))

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')


model_reg = tf.keras.models.load_model('model/regression_model.h5')
model_fashion = tf.keras.models.load_model('model/fashion_mnist_model.h5')
model_food = tf.keras.models.load_model('model/food_1_model.keras')
