import os
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

data_path = os.path.join(".", "datasets", "q4_dataset.mat")
dataset = loadmat(data_path)
inception_features = dataset["inception_features"]
images = dataset["images"]
class_labels = dataset["class_labels"]

# Perform parameter selection from the given space
cs = [1e-6, 1e-4, 1e-2, 1, 1e1, 1e10]
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True)
sets = []
for train_index, test_index in skf.split(inception_features, class_labels):
    X_train, X_test = inception_features[train_index], inception_features[test_index]
    y_train, y_test = class_labels[train_index], class_labels[test_index]
    sets.append((X_train, y_train, X_test, y_test))

# Question 4.1
print("Question 4.1:")

models = [[0] * k] * len(cs)
score = [[0] * k] * len(cs)

# Tune C hyper-parameter of the model
for i, c in enumerate(cs):
    for j in range(k):
        models[i][j] = SVC(kernel='linear', decision_function_shape='ovr', C=c)
        models[i][j].fit(sets[j][0],  (sets[j][1]).ravel())
        score[i][j] = models[i][j].score(
            sets[j % k][2], (sets[j % k][3]).ravel())

for i in range(len(score)):
    print("C =", cs[i])
    for j in range(len(score[i])):
        print("Score of test set", str(j) + ":", score[i][j])
    print()

# Question 4.2
print("Question 4.2:")
cs = [1e-4, 1e-2, 1, 1e1, 1e10]
gammas = [2 ** -4, 2 ** -2, 1, 2 ** 2, 2 ** 10, 'scale']

models = [[[0] * k] * len(gammas)] * len(cs)
score = [[[0] * k] * len(gammas)] * len(cs)

# Tune C and gamma hyper-parameter of the model
for i, c in enumerate(cs):
    for j, g in enumerate(gammas):
        for l in range(k):
            models[i][j][l] = SVC(
                kernel='rbf', decision_function_shape='ovr', C=c, gamma=g)
            models[i][j][l].fit(sets[l][0],  (sets[l][1]).ravel())
            score[i][j][l] = models[i][j][l].score(
                sets[l % k][2], (sets[l % k][3]).ravel())

for i in range(len(score)):
    for j in range(len(score[i])):
        print("C =", cs[i])
        print("Gamma =", gammas[j])
        for k in range(len(score[i][j])):
            print("Score of test set", str(k) + ":", score[i][j][k])
        print()
