"""
Name: Nguyen Van Hoang
Class: Computer Science
MSSV: 21025029

You should understand your code

Several works can be added:
- Improve accuracy
- Improve speed: parrallel processing
- Add more parameters
"""

import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def add_noise_data(input_data, input_labels, n_points, mean, scale):
    """
    Create a noise verstion of the input data

    Params:
        input_data: base input data
        input_labels: base input labels
        n_points: the number of needed points
        mean, scale: the gaussian data
    """
    raw_X = []
    raw_labels = []

    noise = np.random.normal(loc=mean, scale=scale, size=(n_points, 2))
    for i in range(n_points):
        k = np.random.randint(len(input_data))

        x1 = input_data[k][0] + noise[i][0]
        x2 = input_data[k][1] + noise[i][1]

        # We add more difficult for decision tree
        raw_X.append([x1 + x2, x1 * x2,
                      math.sin(x1), 1 / (1 + math.exp(-x2)), x1 / abs(x2) + 1e-5])

        raw_labels.append(input_labels[k])

    return np.array(raw_X), np.array(raw_labels)


if __name__ == "__main__":
    np.random.seed(1)

    n_train = 10000
    std_train = 0.5

    n_test = 1000
    std_test = 0.5

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 0, 0, 1])

    Xtrain, ytrain = add_noise_data(and_X, and_y, n_train, 0., std_train)
    print(Xtrain.shape, ytrain.shape)

    rf = RandomForestClassifier(
        criterion="gini",
        max_depth=5,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(Xtrain, ytrain)

    Xtest, ytest = add_noise_data(and_X, and_y, n_test, 0., std_test)

    output_test = rf.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))

    # xgb = XGBClassifier(max_depth=1)
    # xgb.fit(Xtrain, ytrain)
    # output_test = xgb.predict(Xtest)
    # print("Accuracy (using XGB): ", accuracy_score(ytest, output_test))
    # print("F1 score (using XGB): ", f1_score(ytest, output_test))
