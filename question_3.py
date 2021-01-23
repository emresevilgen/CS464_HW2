import pandas as pd
import os
import numpy as np
import time

train_data_path = os.path.join(".", "datasets", "q3_train_dataset.csv")
train_set = pd.read_csv(train_data_path)

test_data_path = os.path.join(".", "datasets", "q3_test_dataset.csv")
test_set = pd.read_csv(test_data_path)

# Normalize the features
gender = {'male': 1, 'female': 2}
train_set = train_set.applymap(lambda s: gender.get(s) if s in gender else s)
test_set = test_set.applymap(lambda s: gender.get(s) if s in gender else s)

port = {'C': 1, 'Q': 2, 'S': 3}
train_set = train_set.applymap(lambda s: port.get(s) if s in port else s)
test_set = test_set.applymap(lambda s: port.get(s) if s in port else s)

train_set = train_set.astype('float')
test_set = test_set.astype('float')

for col in train_set.columns:
    if (col != 'Survival Status'):
        train_set[col] = (train_set[col] - train_set[col].min()) / \
            (train_set[col].max() - train_set[col].min())
        test_set[col] = (test_set[col] - train_set[col].min()) / \
            (train_set[col].max() - train_set[col].min())


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.batch_size = batch_size

    # Create a list containing mini-batches
    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))

        return mini_batches

    # Logistic regression expression
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Classify all of the test set
    def test_classify(self, test_classify, treshold, weights):
        prob = self.sigmoid(np.dot(test_classify, weights))
        prob[prob >= treshold] = int(1)
        prob[prob < treshold] = int(0)
        return prob.flatten()

    def stochastic_gradient_ascent(self, x, y):
        # Initiale all weights to random numbers drawn from a Gaussian distribution N(0, 0.01)
        mean, sigma = 0, 0.01
        weights = np.random.uniform(mean, sigma, size=(x.shape[1], 1))

        x = np.mat(x)
        y = np.mat(y).reshape((y.shape[0]), 1)

        for i in range(self.num_iter):

            # Create the mini batches
            mini_batches = self.create_mini_batches(x, y, self.batch_size)

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch

                h = self.sigmoid(np.dot(x_mini, weights))

                # Update the weights
                out = y_mini - h
                gradient = np.dot(x_mini.T, out)
                weights = weights + self.learning_rate * gradient

        return weights

    def full_batch_gradient_ascent(self, x, y):
        # Initialize all weights to random numbers drawn from a Gaussian distribution N(0, 0.01)
        mean, sigma = 0, 0.01
        weights = np.random.uniform(mean, sigma, size=(x.shape[1], 1))

        x = np.mat(x)
        y = np.mat(y).reshape((y.shape[0]), 1)

        for i in range(self.num_iter):
            h = self.sigmoid(np.dot(x, weights))

            # Update
            out = y - h
            gradient = np.dot(x.T, out)
            weights = weights + self.learning_rate * gradient

            # Print model weights in each iteration i = 100, 200, ..., 1000
            if i % 100 == 0:
                temp = weights.copy()
                print('Weights: ', temp.reshape((temp.shape[0],)))

        return weights

    # Report class based accuracies
    def report(self, predictions, label):
        predictions = predictions.reshape(-1, 1)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(predictions)):
            if label[i] == predictions[i] and label[i] == 1:
                tp += 1
            elif predictions[i] == 1 and label[i] != predictions[i]:
                fp += 1
            elif label[i] == predictions[i] and label[i] == 0:
                tn += 1
            elif predictions[i] == 0 and label[i] != predictions[i]:
                fn += 1

        accuracy = (tp + tn) / (fp + fn + tp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        npv = tn / (tn + fn)
        fpr = fp / (fp + tn)
        fdr = fp / (fp + tp)
        f1_score = 2 * (recall * precision) / (recall + precision)
        f2_score = (5 * precision * recall) / (4 * precision * recall)
        conf_mat = [[tp, fp], [fn, tn]]

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("NPV:", npv)
        print("FPR:", fpr)
        print("FDR:", fdr)
        print("F1 Score:", f1_score)
        print("F2 Score:", f2_score)
        print("Confusion Matrix:", conf_mat)


train_labels = train_set['Survival Status']
train_set = train_set.drop(columns=['Survival Status'])

test_labels = test_set['Survival Status']
test_set = test_set.drop(columns=['Survival Status'])

# Question 3.1
print("Question 3.1:")
# Choose the learning rate from the following set: 1e-4, 1e-3, 1e-2
learning_rates = [1e-4, 1e-3, 1e-2]

# Perform 1000 iterations to train your model
model_1 = LogisticRegression(
    learning_rate=learning_rates[2], num_iter=1000, batch_size=32)

start = time.time()
weights = model_1.stochastic_gradient_ascent(train_set, train_labels)
end = time.time()
print("Total time elapsed for Question 3.1:", end-start, "seconds")

predictions = model_1.test_classify(test_set, 0.5, weights)

# Report class based accuracies
accuracy = model_1.report(predictions, test_labels)

print()

# Question 3.2
print("Question 3.2:")
# Perform 1000 iterations to train your model
model_2 = LogisticRegression(
    learning_rate=learning_rates[2], num_iter=1000)

start = time.time()
weights = model_2.full_batch_gradient_ascent(train_set, train_labels)
end = time.time()
print("Total time elapsed for Question 3.2:", end-start, "seconds")

predictions = model_2.test_classify(test_set, 0.5, weights)

# Report class based accuracies
accuracy = model_2.report(predictions, test_labels)
