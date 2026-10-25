import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, 't', add_intercept=True)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, t_train)

    # Make sure to save outputs to pred_path_c
    x_test, t_test = util.load_dataset(test_path, 't', add_intercept=True)
    util.plot(x_test, t_test, log_reg.theta, pred_path_c.replace('.txt', '.png'))

    np.savetxt(pred_path_c, log_reg.predict(x_test))

    # Part (d): Train on y-labels and test on true labels
    _, y_train = util.load_dataset(train_path)
    log_reg.fit(x_train, y_train)
    _, y_test = util.load_dataset(test_path)
    util.plot(x_test, y_test, log_reg.theta, pred_path_d.replace('txt', 'png'))

    # Make sure to save outputs to pred_path_d
    np.savetxt(pred_path_d, log_reg.predict(x_test))


    # Part (e): Apply correction factor using validation set and test on true labels
    def h(x):
        return 1 / (1 + np.exp(-x))

    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    h_valid = h(x_valid @ log_reg.theta)
    alpha = np.mean(h_valid[y_valid == 1])

    p_t_given_x = h_valid / alpha
    pred = (p_t_given_x >= 0.5).astype(int)

    theta_e = log_reg.theta.copy()
    c = np.log((alpha / 2) / (1 - alpha / 2))
    theta_e[0] -= c
    
    # Plot and use np.savetxt to save outputs to pred_path_e
    util.plot(x_test, y_test, theta_e, pred_path_e.replace('txt', 'png'))
    np.savetxt(pred_path_e, pred)
    # *** END CODER HERE

