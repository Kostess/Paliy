import numpy as np
from sklearn.preprocessing import StandardScaler

from Neuro import SingleNeuron

data = np.array([
    [25, 4, 30, 0],  # Кошка
    [60, 20, 40, 1],  # Собака
    [27, 4.5, 28, 0],  # Кошка
    [55, 18, 38, 1],  # Собака
    [26, 4.2, 32, 0],  # Кошка
    [62, 22, 42, 1],  # Собака
    [28, 4.8, 29, 0],  # Кошка
    [58, 21, 41, 1],  # Собака
    [24, 3.9, 31, 0],  # Кошка
    [63, 23, 43, 1],  # Собака
    [29, 5, 27, 0],  # Кошка
    [57, 19, 39, 1],  # Собака
    [23, 3.8, 33, 0],  # Кошка
    [61, 22, 44, 1],  # Собака
    [25, 4.1, 30, 0],  # Кошка
    [59, 20, 40, 1],  # Собака
    [26, 4.3, 29, 0],  # Кошка
    [64, 24, 45, 1],  # Собака
    [27, 4.6, 28, 0],  # Кошка
    [56, 18, 37, 1]   # Собака
])

X = data[:, :-1]
y = data[:, -1]

scaler_x = StandardScaler().fit_transform(X)

# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(scaler_x, y, epochs=100, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')