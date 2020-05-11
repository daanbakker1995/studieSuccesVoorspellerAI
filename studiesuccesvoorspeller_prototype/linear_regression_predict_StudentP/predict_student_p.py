from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np  # import for easy use of n-dimensional arrays
import pandas as pd  # import for use of dataframe

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

import tensorflow as tf

# Read csv file and only use columns on given indexes
raw_data = pd.read_csv("studentsZonderNull.csv", sep=";")
raw_data = raw_data[
    ["AfstandSchool", "GemToetsCijferEerstePeriode", "Geslacht", "pcp_Regio", "AantalOplVOORICAIngeschreven",
     "VoorOpleidingsNiveau", "Aanwezigheid1ejaar", "EersteToetsCijfer"]
]
# use 80% of data as training dataframe
dftrain = raw_data.sample(frac=0.9, random_state=0)
# use remaining data as evaluation (testing) dataframe
dfeval = raw_data.drop(dftrain.index)
# remove value that needs to be predicted (labels)
y_train = dftrain.pop('GemToetsCijferEerstePeriode')
y_eval = dfeval.pop('GemToetsCijferEerstePeriode')

# Use feature columns to map all possible outputs
CATEGORICAL_COLUMNS = ['Geslacht', 'pcp_Regio', 'VoorOpleidingsNiveau']

NUMERIC_COLUMNS = ['AfstandSchool', 'AantalOplVOORICAIngeschreven', 'Aanwezigheid1ejaar', 'EersteToetsCijfer']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Define how data is broken into batches and epochs to feed to model.


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # function that will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create dataset with dictionary of data and label
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# for classification (binary results (chance of failing/succeeding)
# linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# train linear regression model to predict label value based on feature values
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # training..
result = linear_est.evaluate(eval_input_fn)  # result with predictions

dfeval['GemToetsCijferEerstePeriode'] = y_eval
dfeval['prediction'] = list(linear_est.predict(eval_input_fn))
print(dfeval)
