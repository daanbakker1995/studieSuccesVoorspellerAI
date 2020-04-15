import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Hyper-parameters
input_size = 5
output_size = 1
learning_rate = 0.001

# Train dataset
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

x_train = np.array(data.drop([predict], 1), dtype='float32')
y_train = np.array(data[predict], dtype='float32')

# Test dataset
testdata = pd.read_csv("test_data.csv", sep=";")
testdata = testdata[["G1", "G2", "G3", "studytime", "failures", "absences"]]

x_test = np.array(testdata.drop([predict], 1), dtype='float32')
y_test = np.array(testdata[predict], dtype='float32')

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(len(x_train)):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    output = model(inputs)
    loss = criterion(output, targets)

    # Backward and optimize
    optimizer.zero_grad()  # clears old gradients from the last step
    loss.backward()  # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
    optimizer.step()  # causes the optimizer to take a step based on the gradients of the parameters.

    # if (epoch + 1) % 5 == 0:
    # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, len(x_train), loss.item()))

# Predictions
for test in range(len(x_test)):
    predicted = model(torch.from_numpy(x_test)).detach().numpy()
    print("Predicted / input / actual value : ", "%.1f" % predicted[test], x_test[test], y_test[test])
