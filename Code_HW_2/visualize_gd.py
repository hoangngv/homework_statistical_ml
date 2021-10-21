import matplotlib.pyplot as plt
import numpy as np


def grad(x, y, eta):
    next_x = x - eta * (4*pow(x, 3) + 4*x - 12*x*y - 4*y + 4)
    next_y = y - eta * (20*y - 6*x**2 - 4*x + 4)
    return [round(next_x, 3), round(next_y, 3)]


def cost(x, y):
    return (x**2 - 3*y - 1)**2 + (2*x - y + 1)**2


def myGD(w_init, eta):
    w = [w_init]
    for it in range(10):
        new_w = grad(w[-1][0], w[-1][1], eta)
        if np.linalg.norm(grad(new_w[0], new_w[1], eta)) / len(new_w) < 1e-3:
            break
        w.append(new_w)
        print("Step = {}/10, x{} = {}, cost = {}".format(it + 1, it + 1, new_w, round(cost(new_w[0], new_w[1]), 3)))
    return w


def plot(f_x, eta):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("cost")
    ax.set_title("Gradient descent with learning rate " + str(eta))

    for i in range(len(f_x) - 1):
        try:
            x = [f_x[i][0], f_x[i+1][0]]
            y = [f_x[i][1], f_x[i+1][1]]
            z = [cost(f_x[i]), cost(f_x[i + 1])]
            ax.plot3D(x, y, z, '-ro')
        except:
            break
    plt.show()


if __name__ == "__main__":
    etas = [0.02, 0.05]
    start = [1, 1]
    for eta in etas:
        print("Apply gradient descent with learning rate {}".format(eta))
        w_new = myGD(start, eta)
        plot(w_new, eta)
