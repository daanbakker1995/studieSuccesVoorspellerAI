from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd  # import for use of dataframe
import tensorflow as tf


# Read csv file and only use columns on given indexes
def get_ai_results():
    raw_data = pd.read_csv("./ai/student-mat.csv", sep=";")
    raw_data = raw_data[
        ["school", "sex", "age", "Mjob", "Fjob", "traveltime",
         "studytime", "failures", "absences", "G1", "G2", "G3"]
    ]
    # use 80% of data as training dataframe
    dftrain = raw_data.sample(frac=0.8, random_state=0)
    print(dftrain[:10])
    # use remaining data as evaluation (testing) dataframe
    dfeval = raw_data.drop(dftrain.index)
    # remove value that needs to be predicted (labels)
    y_train = dftrain.pop('G3')
    print(y_train[:10])
    y_eval = dfeval.pop('G3')

    # Use feature columns to map all possible outputs
    CATEGORICAL_COLUMNS = ['school', 'sex', 'Mjob', 'Fjob']

    NUMERIC_COLUMNS = ['age', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    print(feature_columns)

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

    dfeval['G3'] = y_eval
    dfeval['prediction'] = list(linear_est.predict(eval_input_fn))
    print(dfeval)

    return result
