# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt

def plot_data_comparison(path_A, path_B):
    # 使用 ai 辅助生成的画图代码
    # 可以看出，数据集 A 是非线性可分的，而数据集 B 是线性可分的。
    
    # 1. 读取数据 (跳过表头，逗号分隔)
    data_A = np.loadtxt(path_A, delimiter=',', skiprows=1)
    data_B = np.loadtxt(path_B, delimiter=',', skiprows=1)

    # 2. 根据你提供的数据格式：第0列是y，第1,2列是x0,x1
    y_A, X_A = data_A[:, 0], data_A[:, 1:3]
    y_B, X_B = data_B[:, 0], data_B[:, 1:3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 画数据集 A ---
    axes[0].set_title("Dataset A (Non-linearly Separable)", fontsize=14)
    # 注意：这里正类是 1，负类是 -1
    axes[0].scatter(X_A[y_A == 1, 0], X_A[y_A == 1, 1], c='blue', marker='x', label='y=1')
    axes[0].scatter(X_A[y_A == -1, 0], X_A[y_A == -1, 1], c='red', marker='o', facecolors='none', label='y=-1')
    axes[0].set_xlabel('x0')
    axes[0].set_ylabel('x1')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- 画数据集 B ---
    axes[1].set_title("Dataset B (Linearly Separable)", fontsize=14)
    axes[1].scatter(X_B[y_B == 1, 0], X_B[y_B == 1, 1], c='blue', marker='x', label='y=1')
    axes[1].scatter(X_B[y_B == -1, 0], X_B[y_B == -1, 1], c='red', marker='o', facecolors='none', label='y=-1')
    axes[1].set_xlabel('x0')
    axes[1].set_ylabel('x1')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()



def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    plot_data_comparison('../data/ds1_a.csv', '../data/ds1_b.csv')

    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
