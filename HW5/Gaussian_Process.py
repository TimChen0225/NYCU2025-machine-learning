import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def load_data(filepath):
    try:
        data = np.loadtxt(filepath)
        X_train = data[:, 0:1]
        Y_train = data[:, 1]
        print(f"Data loaded from {filepath}: {X_train.shape[0]} samples.")
        return X_train, Y_train
    except Exception as e:
        print(f"loading data failed: {e}")
        return None, None


def rational_quadratic_kernel(X1, X2, params):
    """
    Args:
        X1 (np.array): shape (N, D)
        X2 (np.array): shape (M, D)
        params (dict): 'sigma', 'l', 'alpha'

    Returns:
        np.array: shape (N, M) - kernel matrix
    """
    sigma = params.get("sigma", 1.0)
    l = params.get("l", 1.0)
    alpha = params.get("alpha", 1.0)

    # sqeuclidean: Squared Euclidean Distance
    #  = sum((xi - xj)^2)
    sq_dist = cdist(X1, X2, "sqeuclidean")
    # k(xi, xj) = sigma^2 * (1 + ||xi - xj||^2 / (2 * alpha * l^2))^(-alpha)
    kernel_matrix = sigma**2 * np.power(1 + sq_dist / (2 * alpha * l**2), -alpha)
    return kernel_matrix


def gaussian_process_regression(X_train, Y_train, X_test, kernel_func, kernel_params, beta):
    """
    Args:
        X_train (np.array): shape (N, D)
        Y_train (np.array): shape (N,)
        X_test (np.array): shape (M, D)
        kernel_func (function): function to compute the kernel matrix
        kernel_params (dict): parameters for the kernel function
        beta (float): noise precision
    Returns:
        mu_test (np.array): shape (M,) - predictive mean
        var_test_diag (np.array): shape (M,) - predictive variance (diagonal)
    """
    kernel = kernel_func(X_train, X_train, kernel_params)
    C = kernel + (1.0 / beta) * np.eye(X_train.shape[0])  # K(x_train, x_train) + (1/beta)I
    K_S = kernel_func(X_train, X_test, kernel_params)  # K(x_train, x_test)
    K_SS = kernel_func(X_test, X_test, kernel_params) + (1.0 / beta) * np.eye(X_test.shape[0])  # K(x_test, x_test)

    C_inv = np.linalg.inv(C)

    mu_test = K_S.T @ C_inv @ Y_train
    var_test = K_SS - K_S.T @ C_inv @ K_S
    var_test_diag = np.diag(var_test)

    return mu_test, var_test_diag


def negative_marginal_log_likelihood(params_array, X_train, Y_train, beta, kernel_func):
    """
    Args:
        params_array (np.array): shape (3,) - [sigma, l, alpha] for optimizer
        X_train (np.array): shape (N, D)
        Y_train (np.array): shape (N,)
        beta (float): noise precision
        kernel_func (function): function to compute the kernel matrix
    Returns:
        float: negative marginal log likelihood
    """
    params = {"sigma": params_array[0], "l": params_array[1], "alpha": params_array[2]}

    K = kernel_func(X_train, X_train, params)
    C = K + (1.0 / beta) * np.eye(X_train.shape[0])
    C_inv = np.linalg.inv(C)
    # L = -1/2 * Y^T * C^-1 * Y - 1/2 * log|C| - N/2 * log(2pi)
    L = -0.5 * Y_train.T @ C_inv @ Y_train - 0.5 * np.log(np.linalg.det(C)) - 0.5 * X_train.shape[0] * np.log(2 * np.pi)
    return -L


def optimize_kernel_params(X_train, Y_train, initial_params, beta):
    """
    Args:
        X_train (np.array): shape (N, D)
        Y_train (np.array): shape (N,)
        initial_params (np.array): shape (3,) - initial [sigma, l, alpha]
        beta (float): noise precision
    Returns:
        np.array: shape (3,) - optimized [sigma, l, alpha]
    """
    print("Starting kernel parameter optimization...")
    optimal_params = minimize(
        fun=negative_marginal_log_likelihood,
        x0=initial_params,
        args=(X_train, Y_train, beta, rational_quadratic_kernel),
        bounds=[(1e-3, None), (1e-3, None), (1e-3, None)],
        options={"maxiter": 1000},
    )
    return optimal_params.x


def plot_gp_result(X_train, Y_train, X_test, mu, var, title, output_filename):
    print(f"start plotting: {output_filename}")

    X_test_1d = X_test.ravel()

    # 95% confidence interval: mu ± 1.96 * std_dev
    std_dev = np.sqrt(var)
    confidence_95_upper = mu + 1.96 * std_dev
    confidence_95_lower = mu - 1.96 * std_dev

    plt.figure(figsize=(12, 8))

    # draw data points
    plt.scatter(X_train, Y_train, c="red", marker="x", label="Training Data")

    # draw mean prediction line
    plt.plot(X_test_1d, mu, "b-", label="Predictive Mean")

    # draw confidence interval
    plt.fill_between(
        X_test_1d, confidence_95_lower, confidence_95_upper, color="blue", alpha=0.2, label="95% Confidence Interval"
    )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("f(X)")
    plt.legend(loc="upper left")
    plt.grid(True)

    # save the plot to file
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Gaussian Process Regression for ML HW5")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./ML_HW05/data/input.data",
        help="Path to the input data file (default: input.data)",
    )
    parser.add_argument(
        "--beta", type=float, default=5.0, help=f"Noise precision beta (default: 5.0, as in spec [cite: 8])"
    )
    parser.add_argument(
        "--plot_start", type=float, default=-60.0, help="Start of the plot range (default: -60.0 [cite: 14])"
    )
    parser.add_argument("--plot_end", type=float, default=60.0, help="End of the plot range (default: 60.0 [cite: 14])")
    parser.add_argument(
        "--plot_points", type=int, default=200, help="Number of points for plotting the prediction line (default: 200)"
    )
    parser.add_argument(
        "--output_task1",
        type=str,
        default="gp_task1.png",
        help="Output filename for Task 1 plot (default: gp_task1.png)",
    )
    parser.add_argument(
        "--output_task2",
        type=str,
        default="gp_task2.png",
        help="Output filename for Task 2 plot (default: gp_task2.png)",
    )

    args = parser.parse_args()
    print(f"using args:\n{args}\n")
    X_train, Y_train = load_data(args.input_file)
    if X_train is None:
        return

    X_test = np.linspace(args.plot_start, args.plot_end, args.plot_points).reshape(-1, 1)

    # --- task 1: using initial parameters ---
    print("\n--- Task 1: using initial parameters ---")
    initial_params_dict = {"sigma": 1, "l": 1, "alpha": 1}
    mu_task1, var_task1 = gaussian_process_regression(
        X_train, Y_train, X_test, rational_quadratic_kernel, initial_params_dict, args.beta
    )
    plot_gp_result(
        X_train, Y_train, X_test, mu_task1, var_task1, "Task 1: GP with Initial Parameters", args.output_task1
    )

    # --- Task 2: optimize params ---
    print("\n--- Task 2: optimize parameters ---")
    initial_params_array = np.array([1.0, 1.0, 1.0])
    print(
        f"inital params: sigma={initial_params_array[0]}, l={initial_params_array[1]}, alpha={initial_params_array[2]}"
    )
    optimal_params_array = optimize_kernel_params(X_train, Y_train, initial_params_array, args.beta)
    optimal_params_dict = {
        "sigma": optimal_params_array[0],
        "l": optimal_params_array[1],
        "alpha": optimal_params_array[2],
    }
    print(
        f'optimal params: sigma={optimal_params_dict["sigma"]}, l={optimal_params_dict["l"]}, alpha={optimal_params_dict["alpha"]}'
    )

    mu_task2, var_task2 = gaussian_process_regression(
        X_train, Y_train, X_test, rational_quadratic_kernel, optimal_params_dict, args.beta
    )
    plot_gp_result(
        X_train, Y_train, X_test, mu_task2, var_task2, "Task 2: GP with Optimized Parameters", args.output_task2
    )


if __name__ == "__main__":
    main()
