# Tensorflow 2.0 Demo

Om de leercurve en de bruikbaarheid van Tensorflow te evalueren is de bijgesloten demo opgesteld.

Uit het literatuuronderzoek is geconcludeerd dat het team voornamelijk gebruik zal maken van het lineaire regressie algoritme<sup>1</sup>.
Dit algoritme wordt gebruikt om op basis van onafhankelijke variabelen (input) een afhankelijke variabele (output) te voorspellen. In de bijgesloten [dataset](student-mat.csv) zal het gaan om 'G3' (grade 3).

### Installatie
Om de demo uit te voeren worden eerst onderstaande packages geinstalleerd:
* Pip (package installer, inbegrepen in Python)<sup>2</sup>
* Tensorflow (framework)<sup>3</sup>
* Pandas (voor gebruik van dataframe)<sup>4</sup>

Ten slotte kan het project geopend worden in een IDE naar wens.

### Uitleg
In onderstaande code wordt de data ingelezen en enkel de variabelen overgenomen waarmee we de output willen voorspellen.

```
raw_data = pd.read_csv("student-mat.csv", sep=";")
raw_data = raw_data[
    ["school", "sex", "age", "Mjob", "Fjob", "traveltime",
     "studytime", "failures", "absences", "G1", "G2", "G3"]
]
```
Nu we onze data hebben delen we het in 2 sets. Trainingsdata om het model te laten leren. En evaluatiedata om het getrainde model te testen.
De waarde die we willen voorspellen scheiden we van beide sets.
```
dftrain = raw_data.sample(frac=0.8, random_state=0)
dfeval = raw_data.drop(dftrain.index)
y_train = dftrain.pop('G3')
y_eval = dfeval.pop('G3')
```
Om het model een beeld te geven van alle mogelijke waarden definieren we categorische en numerieke kolommen. We voegen deze vervolgens toe aan de feature kolommen<sup>5</sup>.
```
CATEGORICAL_COLUMNS = ['school', 'sex', 'Mjob', 'Fjob']
NUMERIC_COLUMNS = ['age', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
```
Om het model te trainen verwacht de Estimator<sup>6</sup> een input_fn als parameter. In onderstaande functie wordt gedefinieerd hoe de data verdeeld wordt in batches en epochs om aan het model te voeren.
```
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # function that will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create dataset with dictionary of data and label
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
```
```
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```
LinearRegressor is de Estimator die het model traint en evalueert. Na het trainen kan het resultaat weergegeven worden.
```
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
```

![Capture](https://github.com/daanbakker1995/studieSuccesVoorspellerAI/blob/master/tensorflow/Capture.JPG?raw=true "Console result")

*Figuur 1 - Console result*


<sup>1</sup> https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86 <br>
<sup>2</sup> https://pypi.org/project/pip/ <br>
<sup>3</sup> https://www.tensorflow.org/install/pip <br>
<sup>4</sup> https://pypi.org/project/pandas/
<sup>5</sup> https://www.tensorflow.org/tutorials/structured_data/feature_columns
<sup>6</sup> https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator