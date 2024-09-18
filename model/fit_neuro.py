import numpy as np
from Neuro import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[-2, -1, 0],
              [25, 6, 1],
              [17, 4, 0],
              [-15, -6, 1]])
y = np.array([0, 1, 1, 0])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')