import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import cvxpy as cp
import numpy as np

def generate_data(plot: bool = False) -> tuple:
    """
    Generate a dataset for SVM classification.

    Parameters:
        plot (bool, optional): Whether to plot the generated dataset. Defaults to False.

    Returns:
        tuple: A tuple containing the generated feature matrix `x` of shape (n_samples, n_features) and the target vector `y` of shape (n_samples,).
    """
    x, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      cluster_std=2.0)
    y[y == 0] = -1

    if plot:
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm_r')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    return x, y

def apply_svm(x: np.ndarray, y: np.ndarray, 
              plt_decision_boundary: bool = False) -> tuple:
    """
    Applies Support Vector Machine (SVM) algorithm to a given dataset.

    Args:
        x: The input feature matrix of shape (n_samples, n_features).
        y: The target vector of shape (n_samples,) containing the class labels.
        plt_decision_boundary: A boolean flag indicating whether to plot the decision boundary and support vectors. Default is False.

    Returns:
        A tuple containing:
        - optimal_w: The optimal weight vector w that defines the hyperplane.
        - optimal_b: The optimal bias term b that defines the hyperplane.
        - prob_status: The status of the problem solution, which can be 'optimal', 'infeasible', 'unbounded', or 'infeasible or unbounded'.
    """
    w = cp.Variable(x.shape[1])
    b = cp.Variable()

    objective = cp.Minimize(0.5 * cp.square(cp.norm(w)))
    constraints = [cp.multiply(y, x @ w + b) - 1 >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    prob_status = problem.status
    optimal_w = w.value
    optimal_b = b.value

    if optimal_w is None: # problem infeasible
        return None, None, prob_status

    if plt_decision_boundary:
        slope = -optimal_w[0] / optimal_w[1]
        intercept = -optimal_b / optimal_w[1]
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        decision_values = (x @ w + b).value * y
        support_vector_indices = np.where(np.abs(decision_values - 1) <= 1e-9)[0] 

        plt.scatter(x[support_vector_indices, 0], x[support_vector_indices, 1], s=95, 
                    facecolors='none', edgecolors='k')
        
        direction_perpendicular = np.array([optimal_w[1], -optimal_w[0]])
        direction_perpendicular /= np.linalg.norm(direction_perpendicular)
        for i in support_vector_indices:
            point = x[i]
            point1 = point + direction_perpendicular * 8
            point2 = point - direction_perpendicular * 8
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'g', linewidth=0.8)

        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm_r')
        plt.plot([x_min, x_max], [x_min * slope + intercept, x_max * slope + intercept], 'k')
        plt.show()

    return optimal_w, optimal_b, prob_status


x, y = generate_data(plot=False)
optimal_w, optimal_b, prob_status = apply_svm(x, y, plt_decision_boundary=True)
if prob_status == 'optimal':
    print(f'Optimal solution reached.')
    print("Optimal w: ", optimal_w)
    print("Optimal b: ", optimal_b)
else:
    print(f'Optimal solution not reached. Status: {prob_status}')
