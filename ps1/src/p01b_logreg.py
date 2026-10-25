import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_eval, y_eval, clf.theta, save_path=pred_path.replace('_pred', '').replace('.txt', '.png'))

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, clf.predict(x_eval))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(theta, x):
            z = np.dot(x, theta)
            return 1 / (1 + np.exp(-z))

        def gradient(theta, x, y):
            m = y.shape[0]
            return 1/m * np.dot(x.T, h(theta, x) - y)

        def hessian(theta, x, y):
            m = y.shape[0]
            h_theta_x = np.reshape(h(theta, x), (-1, 1))
            return 1/m * np.dot(x.T, h_theta_x * (1-h_theta_x) * x)

        def next_theta(theta, x, y):
            theta = theta - np.dot(np.linalg.inv(hessian(theta, x, y)), gradient(theta, x, y))
            return theta

        m, n = x.shape

        self.theta = np.zeros(n)

        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)

        while np.linalg.norm(new_theta - old_theta, 1) >= 1e-5:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)

        self.theta = new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return x @ self.theta >= 0
        # *** END CODE HERE ***
