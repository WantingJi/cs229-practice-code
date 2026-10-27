import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    mse_valid_list = [] # 记录不同 tau 值的 mse

    # 对不同 tau 求 mse
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_valid_pred = clf.predict(x_valid)

        mse = np.mean((y_valid_pred - y_valid) ** 2)
        mse_valid_list.append(mse)

        print(f"MSE with tau {tau}: {mse}")

        # 为每个tau画图（借助 ai）
        plt.figure(figsize=(6, 4))
        plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='Train')
        plt.scatter(x_valid[:, 1], y_valid, c='r', marker='o', facecolors='none', label='Valid')

        sort_idx = np.argsort(x_valid[:, 1])
        plt.plot(x_valid[sort_idx, 1], y_valid_pred[sort_idx], 'g-', label=f'LWR tau={tau}')
        plt.title(f'Validation MSE: {mse:.4f}')
        plt.legend()
        plt.savefig(f'output/p05c_tau_{tau}.png')
        plt.show()

    # 找 best tau
    best_idx = np.argmin(mse_valid_list) # 找最小mse的索引
    best_tau = tau_values[best_idx]

    print(f"\nBest tau: {best_tau}")

    # Fit a LWR model with the best tau value
    best_clf = LocallyWeightedLinearRegression(best_tau)
    best_clf.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    y_test_pred = best_clf.predict(x_test)
    mse_test = np.mean((y_test_pred - y_test) ** 2)
    print(f"Test MSE with best tau {best_tau}: {mse_test}")

    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred)
    
    # Plot data
    # 参考 ai 给的代码
    plt.figure(figsize=(8, 5))
    # 画出训练集数据（作为背景参考，可选）
    plt.scatter(x_train[:, 1], y_train, c='b', marker='x', alpha=0.3, label='Training Set')
    
    # 画出测试集数据（真实标签）
    plt.scatter(x_test[:, 1], y_test, c='r', marker='o', facecolors='none', label='Test Set (True Labels)')
    
    # 画出测试集上的预测曲线
    # 注意：必须把 x_test 排序，否则线会乱连
    sort_idx_test = np.argsort(x_test[:, 1])
    x_test_sorted = x_test[sort_idx_test, 1]
    y_test_pred_sorted = y_test_pred[sort_idx_test]
    
    plt.plot(x_test_sorted, y_test_pred_sorted, 'g-', linewidth=2, label=f'LWR Prediction (tau={best_tau})')
    
    plt.title(f'Test Set Predictions (Best tau={best_tau}, Test MSE: {mse_test:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'output/p05c_best_test_tau_{best_tau}.png')
    plt.show()

    # *** END CODE HERE ***
