import pandas as pd
import numpy as np
import os

# Question 2.2
print("Question 2.2:")

data_path = os.path.join(".", "datasets", "q2_dataset.csv")
x = pd.read_csv(data_path)

# Normalize the features by min max scaling
for col in x.columns:
    if (col != 'Chance of Admission'):
        x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())

# First, shuffle and split the dataset to 5 folds where each fold contains 100 samples
no_of_folds = 5
x = x.sample(frac=1)
xs = np.array_split(x, no_of_folds)
ys = [0] * no_of_folds
for i in range(len(xs)):
    xs[i] = xs[i].reset_index(drop=True)
    ys[i] = xs[i]['Chance of Admission']
    xs[i] = xs[i].drop(columns=['Chance of Admission'])

# Separate fold i as test set and train a model on the remaining folds.
# In the end, you should have trained 5 models, one per test fold.
models = [0] * no_of_folds
predictions = [0] * no_of_folds
truths = [0] * no_of_folds

for i in range(len(xs)):
    test_set = xs[i]
    truths[i] = ys[i]

    train_set = []
    train_output = []

    for j in range(len(xs)):
        if i != j:
            if len(train_set) == 0:
                train_set = xs[j].copy()
                train_output = ys[j].copy()
            else:
                train_set.append(xs[j], ignore_index=True)
                train_output.append(ys[j], ignore_index=True)

    train_set = train_set.to_numpy()
    train_set_t = train_set.transpose()
    model = train_set_t.dot(train_set)
    model = np.linalg.inv(model)
    model = model.dot(train_set_t)
    model = model.dot(train_output)
    models[i] = model

    # Get the predictions for the test sets
    predictions[i] = test_set.dot(model)


def r_square(predictions, truths):
    mean = truths.mean()
    SS_tot = sum((predictions - mean).pow(2))
    SS_res = sum((predictions - truths).pow(2))
    return 1 - SS_res / SS_tot


def mse(predictions, truths):
    return sum((predictions - truths).pow(2)) / len(predictions)


def mae(predictions, truths):
    return sum((predictions - truths).abs()) / len(predictions)


def mape(predictions, truths):
    return sum(((predictions - truths) / truths).abs()) / len(predictions)


# For each model, calculate R2, mean squared error (MSE),
# mean absolute error (MAE) and mean absolute percentage error (MAPE).
for i in range(len(predictions)):
    print("R^2 of model", str(i+1) + ":", r_square(predictions[i], truths[i]))
    print("MSE of model", str(i+1) + ":", mse(predictions[i], truths[i]))
    print("MAE of model", str(i+1) + ":", mae(predictions[i], truths[i]))
    print("MAPE of model", str(i+1) + ":", mape(predictions[i], truths[i]))
    print()

# Question 2.3
print("Question 2.3:")



