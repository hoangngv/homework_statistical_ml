"""
Name: Tran Minh Duc
MSSV: 21025026

You should understand your code

Several works can be added:
- Improve accuracy
- Improve speed: parrallel processing
- Add more parameters
"""

import matplotlib.pyplot as plt
import numpy as np

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

        #raw_X.append([x1 + x2, x1*x2,
                      #math.sin(x1), 1/(1 + math.exp(-x2)), x1/abs(x2) + 1e-5])
        raw_X.append([x1, x2])

        raw_labels.append(input_labels[k])

    return np.array(raw_X), np.array(raw_labels)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter

    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def train(self, X, y):
        X = self._add_intercept(X)
        losses = []

        self._weights = np.zeros(X.shape[1])
        for i in range(self.num_iter):
            z = np.dot(X, self._weights)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self._weights -= self.lr * gradient

            if (i % 1000 == 0):
                z = np.dot(X, self._weights)
                h = sigmoid(z)
                print(f'loss: {self._loss(h, y)} \t')
                losses.append(self._loss(h, y))

        return losses

    def predict(self, X):
        X = self._add_intercept(X)
        return sigmoid(np.dot(X, self._weights))


if __name__ == "__main__":
    np.random.seed(1)

    n_train = 10000
    std_train = 0.1

    n_test = 1000
    std_test = 0.5

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 0, 0, 1])

    Xtrain, ytrain = add_noise_data(and_X, and_y, n_train, 0., std_train)
    Xtest, ytest = add_noise_data(and_X, and_y, n_test, 0., std_test)
    print(Xtrain.shape, ytrain.shape)
    print(Xtest.shape, ytest.shape)

    clf = LogisticRegression(num_iter=100000)
    losses=clf.train(Xtrain, ytrain)
    plt.plot(np.array(range(len(losses)))*1000, losses)
    plt.show()

    plt.scatter(Xtrain[ytrain==0][:,0], Xtrain[ytrain==0][:,1], label='0train')
    plt.scatter(Xtrain[ytrain==1][:,0], Xtrain[ytrain==1][:,1], label='1trainn')

    logits = clf.predict(Xtrain)
    preds = logits > 0.5
    acc = sum(preds == ytrain) / len(preds)
    print('trainacc', acc)

    logits = clf.predict(Xtest)
    preds = logits > 0.5
    acc = sum(preds == ytest) / len(preds)
    print('testacc', acc)

    w = clf._weights[1:]
    a = -w[0] / w[1]
    xx = np.linspace(-1, 2)
    yy = a * xx - (clf._weights[0]) / w[1]

    plt.scatter(Xtest[ytest==0][:,0], Xtest[ytest==0][:,1], label='0test')
    plt.scatter(Xtest[ytest==1][:,0], Xtest[ytest==1][:,1], label='1test')
    plt.plot(xx, yy, 'k-')
    plt.legend()
    plt.show()