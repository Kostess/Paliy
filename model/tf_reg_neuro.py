import tensorflow as tf
import numpy as np

realty = np.array([
    [120, 3, 10],
    [90, 2, 5],
    [150, 4, 15],
    [80, 2, 3],
    [200, 5, 20],
    [110, 3, 8],
    [130, 4, 12],
    [70, 2, 2],
    [180, 4, 18],
    [100, 3, 6],
    [140, 4, 14],
    [95, 3, 4],
    [160, 5, 16],
    [75, 2, 2.5],
    [170, 4, 17],
    [105, 3, 7],
    [125, 4, 11],
    [85, 2, 3.5],
    [190, 5, 19],
    [115, 3, 9]
], dtype=float)
price = np.array([
    5000, 3500, 6000, 3000, 7500, 4500, 5500, 2800, 7000, 4000,
    5800, 3800, 6500, 2900, 6800, 4200, 5200, 3200, 7200, 4700
], dtype=float)

model_reg_neuro = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3, ),
    tf.keras.layers.Dense(1, activation='linear')  # Один выход для регрессии
])

model_reg_neuro.compile(optimizer='adam', loss='mean_squared_error')

model_reg_neuro.fit(realty, price, epochs=500)

# Сохранение модели для регрессии
model_reg_neuro.save('regression_model.h5')
