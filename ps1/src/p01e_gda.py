import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    # Train a GDA classifier
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_eval, y_eval, clf.theta, save_path=pred_path.replace('_pred', '').replace('.txt', '.png'))

    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, clf.predict(x_eval))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m, n = x.shape

        phi = np.sum(y) / m
        mu_0 = np.dot(x.T, 1 - y) / np.sum(1 - y)
        mu_1 = np.dot(x.T, y) / np.sum(y)

        y_reshaped = np.reshape(y, (m, -1))
        mu_y = y_reshaped * mu_1 + (1 - y_reshaped) * mu_0

        sigma = np.dot((x - mu_y).T, x - mu_y) / m
        sigma_inv = np.linalg.inv(sigma)
        
        # Write theta in terms of the parameters    
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = 1/2 * (mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1) + np.log(phi / (1 - phi))

        self.theta = np.insert(theta, 0, theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x_i = np.hstack([np.ones((x.shape[0], 1)), x])
        return x_i @ self.theta >= 0
        # *** END CODE HERE
