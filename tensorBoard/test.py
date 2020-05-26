import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn import linear_model

import os
from datetime import datetime

model = tf.keras.Sequential([
    keras.layers.Dense(units=2, input_shape=[2])])
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

xs = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=float)
ys = np.array([[-3.0, -3.0], [-1.0, -1.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]], dtype=float)  # 2*x-1

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xs, ys, test_size=0.1)

log_dir = ".\\tensorflow_logs\\test\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train,
          y_train,
          epochs=15,
          verbose=0,
          callbacks=[tensorboard_callback])

print(model.predict(x_test))
