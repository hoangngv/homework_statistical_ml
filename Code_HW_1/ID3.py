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

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

label_yes = "Yes"
label_no = "No"


class Node:

    def __init__(self, feature, value, left, right, depth, label):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.depth = depth


class RandomTree:

    def __init__(self, n_depth=6, n_random_features=0.1, seed=1):
        self.n_depth = n_depth
        self.n_random_features = n_random_features
        self.rng = np.random.RandomState(seed)
        self.tree = []

    def _fit(self, at, X, y):

        # If we reach n_depth
        if self.tree[at].depth >= self.n_depth:
            n_label0 = len(y[y == label_no])
            n_label1 = len(y[y == label_yes])
            if n_label0 < n_label1:
                self.tree[at].label = label_yes
            else:
                self.tree[at].label = label_no
            return

        # find the best split
        best_ft = -1
        best_value = None
        best_score = -1.
        entropy = calc_entropy(y)
        self.tree[at].entropy = entropy

        for feature_idx in range(X.shape[1]):
            [ft_best_score, ft_best_value, ft_left_entropy,
             ft_right_entropy] = compute_information_gain(X, y, feature_idx)

            if ft_best_score > best_score:
                best_ft = feature_idx  # best feature
                best_value = ft_best_value
                best_score = ft_best_score
                self.tree[at].left_entropy = ft_left_entropy
                self.tree[at].right_entropy = ft_right_entropy

        self.tree[at].feature = best_ft
        self.tree[at].value = best_value
        self.tree[at].left = len(self.tree)
        self.tree[at].right = len(self.tree) + 1

        left_node = Node(-1, None, -1, -1, self.tree[at].depth+1, label_yes)
        right_node = Node(-1, None, -1, -1, self.tree[at].depth+1, label_yes)

        self.tree += [left_node]
        self.tree += [right_node]

        mask = X[:, best_ft] == best_value
        x_left = X[mask]
        y_left = y[mask]

        mask = X[:, best_ft] != best_value
        x_right = X[mask]
        y_right = y[mask]

        self._fit(self.tree[at].left, x_left, y_left)
        self._fit(self.tree[at].right, x_right, y_right)

    def fit(self, X, y):
        node = Node(-1, None, -1, -1, 0, label_yes)
        self.tree += [node]
        self._fit(0, X, y)

    def _predict(self, x):
        at = 0
        while True:
            i = self.tree[at].feature
            value = self.tree[at].value

            if i == -1:
                return self.tree[at].label

            if x[i] == value:
                at = self.tree[at].left
            else:
                at = self.tree[at].right

    def predict(self, X):
        ret = [self._predict(x) for x in X]
        return np.array(ret)


def calc_entropy(y):
    if len(y) <= 1:
        return 0

    n_label0 = len(y[y == label_no])
    p = 1.*n_label0/len(y)
    if (p < 1e-5) or abs(1-p) < 1e-5:
        return 0

    return -p*math.log(p) - (1-p)*math.log(1-p)


# Q2.1
def compute_information_gain(X, y, feature):
    ft_best_score = -1.
    ft_best_value = None
    ft_left_entropy = None
    ft_right_entropy = None
    parent_entropy = calc_entropy(y)
    ft_value_set = set(X[:, feature])

    for ft_value in ft_value_set:
        y_left = y[X[:, feature] == ft_value]
        y_right = y[X[:, feature] != ft_value]

        child_entropy = (len(y_left)/len(y)) * calc_entropy(y_left) + (1-len(y_left)/len(y)) * calc_entropy(
            y_right)
        information_gain = parent_entropy - child_entropy

        if information_gain > ft_best_score:
            ft_best_score = information_gain
            ft_best_value = ft_value
            ft_left_entropy = calc_entropy(y_left)
            ft_right_entropy = calc_entropy(y_right)

    return [ft_best_score, ft_best_value, ft_left_entropy, ft_right_entropy]


# Build decision tree on X and y
# List of:
# node_index, node_feature[0..3], (feature_value -> child_index) : internal node
# leafnode: node_index, node_features = -1, Yes/No
def build_ID3(X, y):
    rf = RandomTree(n_depth=4)
    rf.fit(X, y)
    return rf


if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target = ['PlayTennis']
    X_train = df[features].values
    y_train = df[target].values
    le = LabelEncoder()
    le.fit(y_train.reshape(-1,))

    # Q2.1
    print("Information Gain of Outlook: ", compute_information_gain(X_train, y_train, 0)[0])

    # Q2.2
    rf = build_ID3(X_train, y_train)

    print("Evaluate:")

    test = pd.read_csv("./test.csv")
    X_test = test[features].values
    y_test = test[target].values
    y_test = y_test.reshape(-1,)
    output_test = rf.predict(X_test)
    output_test = le.transform(output_test)  # label encoding
    y_test = le.transform(y_test)

    print("y true = {}, y predict = {}".format(y_test, output_test))
    print("Accuracy: ", accuracy_score(y_test, output_test))
    print("F1 score: ", f1_score(y_test, output_test))
