import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import shap

label_encoder = LabelEncoder()

# prs_PersoonsID[0] is seen as useless datatype since it is unique.
CATEGORICAL_COLUMNS = ['pcp_Regio', 'isc_OpleidingsCode', 'Geslacht', 'VoorOpleidingsNiveau']
NUMERIC_COLUMNS = ['AfstandSchool', 'LeeftijdMaandenEersteInschr', 'NrStdInEersteKlas',
                   'AantalOplVOORICAIngeschreven', 'Aanwezigheid1ejaar', 'EersteToetsCijfer',
                   'GemToetsCijferEerstePeriode']
df = pd.read_csv('data.csv')

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

model.fit(x_train, y_train, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)

df_train_normed_summary = x_train[:100]
explainer = shap.KernelExplainer(model.predict, df_train_normed_summary)
shap_values = explainer.shap_values(df_train_normed_summary)
shap.summary_plot(shap_values[0], df_train_normed_summary, properties, plot_type="bar")
