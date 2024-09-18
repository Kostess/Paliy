import numpy as np
from Neuro import SingleNeuron
from sklearn.preprocessing import StandardScaler

data = np.array([
    [175, 70, 25, 1],
    [160, 55, 30, 0],
    [180, 80, 22, 1],
    [155, 50, 28, 0],
    [170, 65, 24, 1],
    [165, 60, 32, 0],
    [178, 75, 26, 1],
    [158, 53, 29, 0],
    [182, 85, 23, 1],
    [163, 58, 31, 0],
    [172, 72, 27, 1],
    [167, 63, 33, 0],
    [176, 74, 25, 1],
    [161, 56, 30, 0],
    [181, 82, 22, 1],
    [156, 51, 28, 0],
    [171, 66, 24, 1],
    [166, 61, 32, 0],
    [179, 76, 26, 1],
    [159, 54, 29, 0]
])

X = data[:, :-1]
y = data[:, -1]

# Нормализуем признаки
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X_normalized, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')