from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

raw_data = pd.read_csv("student-mat.csv", sep=";")
raw_data = raw_data[
    ["school", "sex", "age", "Mjob", "Fjob", "traveltime",
     "studytime", "failures", "absences", "G1", "G2", "G3"]
]
dftrain = raw_data.sample(frac=0.8, random_state=0)
print(dftrain[:10])
dfeval = raw_data.drop(dftrain.index)
y_train = dftrain.pop('G3')
print(y_train[:10])
y_eval = dfeval.pop('G3')

CATEGORICAL_COLUMNS = ['school', 'sex', 'Mjob', 'Fjob']

NUMERIC_COLUMNS = ['age', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Voor classificatie evt.
# linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

dfeval['G3'] = y_eval
dfeval['prediction'] = list(linear_est.predict(eval_input_fn))
print(dfeval)