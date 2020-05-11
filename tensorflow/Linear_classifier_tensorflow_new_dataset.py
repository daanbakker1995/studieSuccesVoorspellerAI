import tensorflow as tf
import pandas as pd

# Datafile
RAW_DATA = "studentsZonderNull.csv"
# Read csv file
raw_data = pd.read_csv(RAW_DATA, sep=";")
# use 80% of data as training dataframe
training_set = raw_data.sample(frac=0.8, random_state=0)
# use remaining data as evaluation (testing) dataframe
eval_set = raw_data.drop(training_set.index)

train_column = training_set.pop('GemToetsCijferEerstePeriode')
eval_column = eval_set.pop('GemToetsCijferEerstePeriode')

# set catogorical columns
CATEGORICAL_COLUMNS = ["pcp_Regio", "isc_OpleidingsCode", "Geslacht", "PCertificaat_Opl", "VoorOpleidingsNiveau"]
# set numeric columns
NUMERIC_COLUMNS = ["prs_PersoonsID", "AfstandSchool", "isc_VanDatum", "LeeftijdMaandenEersteInschr",
                   "NrStdInEersteKlas", "AantalOplVOORICAIngeschreven", "PropCertificaatDatum", "HeeftP",
                   "Aanwezigheid1ejaar", "EersteToetsCijfer"]

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = training_set[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_fn(training_set, train_column)
eval_input_fn = make_input_fn(eval_set, eval_column, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])
result = list(linear_est.predict(eval_input_fn))
print(eval_set.loc[0])
print("Evaluation Column: ", eval_column.loc[0])
print("Prediction", result[0])
#
# print(eval_set.loc[5])
# print("Survived: ", eval_column.loc[5])
# print("Survival prediction", result[5]['probabilities'][1])
