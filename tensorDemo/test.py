import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# first we'll read the dataset and seperate it
data = pd.read_csv("student-mat.csv", sep=";")
# since this PoC is merely to demonstrate training a model, we'll only use a couple parameters
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# G3 is the 3rd grade. that's the parameter we'll predict
predict = "G3"

# array of attributes and labels without G3, which we are going to predict
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# splitting arrays into training and testing arrays. 10% test sample
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
# produce best fit line
linear.fit(x_train, y_train)
# calculate accuracy of trainingmodel
acc = linear.score(x_test, y_test)

predictions = linear.predict(x_test)

# print G3 predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])