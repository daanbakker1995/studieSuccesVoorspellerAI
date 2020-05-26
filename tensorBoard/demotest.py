import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import shap
from datetime import datetime

label_encoder = LabelEncoder()
# CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'embark_town', 'alone']
CATEGORICAL_COLUMNS = ['sex', 'class']
NUMERIC_COLUMNS = ['age', 'n_siblings_spouses', 'parch', 'fare']

df = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
properties = list(df.columns.values)
print(properties)
properties.remove('survived')

# Temporarily remove categorical columns
# properties.remove('sex')
# properties.remove('class')
properties.remove('deck')
properties.remove('embark_town')
properties.remove('alone')


def transform_fn(label):
    code = np.array(df[label])
    vec = label_encoder.fit_transform(code)
    tf.keras.utils.to_categorical(vec)
    df[label] = vec


for c in CATEGORICAL_COLUMNS:
    transform_fn(c)

x = df[properties]
y = df['survived']

print(x)

x_np = np.asarray(x, np.float32)
y_np = np.asarray(y, np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

log_dir = ".\\tensorflow_logs\\test\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, verbose=0, validation_split=0.2, callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)

df_train_normed_summary = x_train[:100]
explainer = shap.KernelExplainer(model.predict, df_train_normed_summary)
shap_values = explainer.shap_values(df_train_normed_summary)
shap.summary_plot(shap_values[0], df_train_normed_summary, properties, plot_type="bar")
