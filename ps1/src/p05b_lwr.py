import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LocallyWeightedLinearRegression(tau)
    # Fit a LWR model
    clf.fit(x_train, y_train)
    # Get MSE value on the validation set

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # Plot validation predictions on top of training set
    
    y_pred = clf.predict(x_eval)
    # No need to save predictions
    # Plot data


    # 使用 ai 生成的画图代码
    mse = np.mean((y_eval - y_pred)**2)
    print(f"MSE on validation set: {mse}")

    # 开始画图
    plt.figure(figsize=(8, 5))
    
    # 注意：由于 x_train 和 x_eval 都是带截距的，真实特征在第二列 (索引为 1)
    # 训练集用蓝色 x
    plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='Training Set')
    # 验证集用红色 o
    plt.scatter(x_eval[:, 1], y_eval, c='r', marker='o', facecolors='none', label='Validation Set')
    
    # 局部加权回归预测的是一条曲线，需要按 x 排序后连线
    sort_idx = np.argsort(x_eval[:, 1])
    x_sorted = x_eval[sort_idx, 1]
    y_pred_sorted = y_pred[sort_idx]
    plt.plot(x_sorted, y_pred_sorted, 'g-', linewidth=2, label=f'LWR (tau={tau})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Locally Weighted Linear Regression')
    plt.legend()
    plt.savefig('output/p05b_lwr.png')
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        # 局部加权回归是非参数学习算法，没有全局固定的参数theta
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        # 对每一个需要预测的数据，根据样本集里的数据算权重
        for i in range(m):
            # 对第 i 个数据做局部加权回归
            x_eval = x[i]

            # 算权重值
            diff = self.x - x_eval
            dist_sq = np.sum(diff**2, axis=1)
            w_diag = np.exp(-dist_sq / (2*self.tau**2))
            W = np.diag(w_diag)

            # 根据正规方程求解 theta
            theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y

            # 输出预测值
            y_pred[i] = x_eval @ theta

        return y_pred
        # *** END CODE HERE ***
