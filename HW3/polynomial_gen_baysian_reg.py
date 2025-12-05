#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
import argparse


def polynomial_basis_linear_model_data_generator(n: int, a: float, w: np.ndarray) -> tuple[float, float]:
    """
    生成一個多項式基底線性模型的數據點 (x, y)。[cite: 21]

    Args:
        n (int): 基底的數量 (多項式的最高次數為 n-1)。[cite: 18]
        a (float): 雜訊 e ~ N(0, a) 的變異數。[cite: 15, 18]
        w (np.ndarray): n x 1 的權重向量。[cite: 14, 18]

    Returns:
        tuple[float, float]: 一個數據點 (x, y)。
    """
    x = np.random.uniform(-1.0, 1.0)
    phi_x = np.array([x**i for i in range(n)])
    e = np.random.normal(0, math.sqrt(a))
    y = w.T @ phi_x + e
    return x, y


class BayesianLinearRegression:

    def __init__(self, n: int, a: float, b: float, true_w: np.ndarray):
        """
        Args:
            n (int): 基底的數量。
            a (float): 觀測雜訊的變異數 (variance)。
            b (float): 事前分佈的精確度 (precision)。[cite: 70]
            true_w (np.ndarray): 用於生成數據的真實權重向量。
        """
        self.n = n
        self.a = a
        self.beta = 1.0 / self.a if self.a != 0 else float("inf")
        self.b = b
        self.true_w = true_w

        self.posterior_mean = np.zeros(n)
        self.posterior_covariance = (1.0 / self.b) * np.identity(self.n)

        self.data_points = {"x": [], "y": []}
        self.results_at_10 = {}
        self.results_at_50 = {}

    def run(self, max_iterations: int = 200):
        print(f"Bayesian Linear Regression")
        print(f"n={self.n}, a={self.a}, b={self.b}, w={self.true_w.tolist()}")
        print("-" * 50)

        for i in range(1, max_iterations + 1):
            # get a data point
            x_new, y_new = polynomial_basis_linear_model_data_generator(self.n, self.a, self.true_w)
            self.data_points["x"].append(x_new)
            self.data_points["y"].append(y_new)

            # 上一輪的posterior，這一輪的prior
            prior_mean = self.posterior_mean
            prior_covariance = self.posterior_covariance

            # S_N = (S_N-1 + beta * phi(x_new) * phi(x_new).T)^-1
            phi_new = np.array([x_new**j for j in range(self.n)])
            S_N_inv = np.linalg.inv(prior_covariance) + self.beta * np.outer(phi_new, phi_new)
            self.posterior_covariance = np.linalg.inv(S_N_inv)

            # m_N = S_N * (S_N-1^-1 * m_N-1 + beta * phi(x_new) * y_new)
            term1 = np.linalg.inv(prior_covariance) @ prior_mean
            term2 = self.beta * phi_new * y_new
            self.posterior_mean = self.posterior_covariance @ (term1 + term2)

            predictive_mean = self.posterior_mean.T @ phi_new
            predictive_variance = self.a + phi_new.T @ self.posterior_covariance @ phi_new

            self._print_iteration_results((x_new, y_new), predictive_mean, predictive_variance)

            if i == 10:
                self.results_at_10 = {
                    "mean": np.copy(self.posterior_mean),
                    "cov": np.copy(self.posterior_covariance),
                    "data_points": self.data_points.copy(),
                }
            if i == 50:
                self.results_at_50 = {
                    "mean": np.copy(self.posterior_mean),
                    "cov": np.copy(self.posterior_covariance),
                    "data_points": self.data_points.copy(),
                }

            if i > 1 and np.linalg.norm(self.posterior_mean - prior_mean) < 1e-4:
                print(f"\nConverged after {i} iterations.")
                break

        self.final_results = {"mean": np.copy(self.posterior_mean), "cov": np.copy(self.posterior_covariance)}

    def visualize(self):
        """
        視覺化貝氏線性迴歸的結果。[cite: 78]
        """
        plot_x = np.linspace(-2.0, 2.0, 100)

        ground_truth_y = [self.true_w.T @ np.array([x**i for i in range(self.n)]) for x in plot_x]
        ground_truth_upper = ground_truth_y + np.sqrt(self.a)
        ground_truth_lower = ground_truth_y - np.sqrt(self.a)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def _plot_results(ax, title, posterior_mean, posterior_cov, data_points=None):
            predictive_means = []
            predictive_variances = []
            for x in plot_x:
                phi = np.array([x**i for i in range(self.n)])
                mean = posterior_mean.T @ phi
                var = self.a + phi.T @ posterior_cov @ phi
                predictive_means.append(mean)
                predictive_variances.append(var)

            ax.plot(plot_x, predictive_means, "k")
            upper_bound = np.array(predictive_means) + predictive_variances
            lower_bound = np.array(predictive_means) - predictive_variances
            ax.plot(plot_x, upper_bound, "r")
            ax.plot(plot_x, lower_bound, "r")

            if data_points:
                ax.scatter(data_points["x"], data_points["y"], s=10)

            ax.set_title(title)
            ax.set_ylim(-15, 25)

        ax = axes[0, 0]
        ax.plot(plot_x, ground_truth_y, "k")
        ax.plot(plot_x, ground_truth_upper, "r")
        ax.plot(plot_x, ground_truth_lower, "r")
        ax.set_title("Ground truth")
        ax.set_ylim(-15, 25)

        ax = axes[0, 1]
        _plot_results(ax, "Predict result", self.final_results["mean"], self.final_results["cov"], self.data_points)

        ax = axes[1, 0]
        if self.results_at_10:
            _plot_results(
                ax,
                "After 10 incomes",
                self.results_at_10["mean"],
                self.results_at_10["cov"],
                self.results_at_10["data_points"],
            )
        else:
            ax.set_title("After 10 incomes (not reached)")

        ax = axes[1, 1]
        if self.results_at_50:
            _plot_results(
                ax,
                "After 50 incomes",
                self.results_at_50["mean"],
                self.results_at_50["cov"],
                self.results_at_50["data_points"],
            )
        else:
            ax.set_title("After 50 incomes (not reached)")

        plt.tight_layout()
        plt.show()

    def _print_iteration_results(self, data_point, pred_mean, pred_var):
        """格式化並印出單次迭代的結果。"""
        x, y = data_point
        print(f"Add data point ({x:.5f}, {y:.5f}):\n")
        print("Posterior mean:")
        for val in self.posterior_mean:
            print(f"{val: .8f}")
        print("\nPosterior variance:")
        np.set_printoptions(precision=8, suppress=True)
        print(self.posterior_covariance)
        print(f"\nPredictive distribution ~ N({pred_mean:.5f}, {pred_var:.5f})\n")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Linear Regression with Polynomial Basis Functions")
    parser.add_argument("PRIOR_PRECISION_B", help="PRIOR_PRECISION_B")
    parser.add_argument("N_BASIS", help="N_BASIS")
    parser.add_argument("NOISE_VARIANCE_A", help="NOISE_VARIANCE_A")
    parser.add_argument("TRUE_W", help="TRUE_W")
    args = parser.parse_args()

    N_BASIS = int(args.N_BASIS)
    NOISE_VARIANCE_A = float(args.NOISE_VARIANCE_A)
    PRIOR_PRECISION_B = float(args.PRIOR_PRECISION_B)
    TRUE_W = np.array([float(w) for w in args.TRUE_W.split(",")])

    blr = BayesianLinearRegression(n=N_BASIS, a=NOISE_VARIANCE_A, b=PRIOR_PRECISION_B, true_w=TRUE_W)

    blr.run(max_iterations=50)

    blr.visualize()
