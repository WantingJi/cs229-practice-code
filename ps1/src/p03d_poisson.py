import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    y_pred = clf.predict(x_eval)
    plt.figure(figsize=(8, 6))
    plt.plot(y_eval, 'go', label='True Labels', markersize=6)
    plt.plot(y_pred, 'rx', label='Predictions', markersize=6)
    plt.title('Validation Set: Poisson Regression')
    plt.xlabel('Sample Index')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(pred_path.replace('txt', 'png'))
    # util.plot(x_eval, y_eval, clf.theta, save_path=pred_path.replace('txt', 'png'))
    np.savetxt(pred_path, clf.predict(x_eval))

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        theta = np.zeros(n)
        grad = np.ones(n)

        while np.linalg.norm(grad) >= self.eps:
            lamda = np.exp(x @ theta)
            grad = x.T @ (y - lamda) / m
            theta += self.step_size * grad

        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
