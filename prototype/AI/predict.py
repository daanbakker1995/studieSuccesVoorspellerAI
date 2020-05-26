import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import shap
from datetime import datetime


def get_ai_results():
    label_encoder = LabelEncoder()

    # prs_PersoonsID[0] is seen as useless datatype since it is unique.
    CATEGORICAL_COLUMNS = ['pcp_Regio', 'isc_OpleidingsCode', 'Geslacht', 'VoorOpleidingsNiveau']
    NUMERIC_COLUMNS = ['AfstandSchool', 'LeeftijdMaandenEersteInschr', 'NrStdInEersteKlas',
                       'AantalOplVOORICAIngeschreven', 'Aanwezigheid1ejaar', 'EersteToetsCijfer',
                       'GemToetsCijferEerstePeriode']
    df = pd.read_csv('./AI/data.csv')

    properties = list(df.columns.values)
    properties.remove('prs_PersoonsID')
    properties.remove('isc_VanDatum')
    properties.remove('PCertificaat_Opl')
    properties.remove('PropCertificaatDatum')

    properties.remove('HeeftWisInVooropleiding')
    properties.remove('HeeftBijzOmstandigheden')

    properties.remove('HeeftP')

    def transform_fn(label):
        print(label, ": \n", df[label][:10])
        vec = label_encoder.fit_transform(df[label])
        print(label, ": \n", vec[:10])
        df_num = pd.DataFrame(vec)
        print(label, ": \n", df_num[:10])
        df_num.rename(columns={0: label},
                      inplace=True)
        df[label] = df_num
        print(label, ": \n", df[label][:10])

    for c in CATEGORICAL_COLUMNS:
        transform_fn(c)

    print(df[:10])

    x = df[properties]
    y = df['HeeftP']

    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.3, random_state=0)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(11,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=1,
              validation_split=0.3
              )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(x_test)
    print("prediction: ", predictions[:10])
    print("actual: ", y_test[:10])

    return combine_data(properties, predictions.tolist(), x_test)


def combine_data(properties, predictions, students):
    index = 0
    new_list = []

    for prediction in predictions:
        propertyIndex = 0
        studentValues = {}

        for studentValue in students[index]:
            currentProperty = properties[propertyIndex]
            studentValues[currentProperty] = studentValue

            propertyIndex += 1

        new_list.append({
            "student": studentValues,
            "successPercentage": prediction[0]
        })

        index += 1

    return new_list
