#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=10000, tolerance=1e-5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance  # 用於收斂判斷
        self.w = None  # 權重 (weights)

    def _sigmoid(self, z):
        # np.clip 防止 exp 溢位
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit_gradient_descent(self, X, y):
        """
        使用最陡梯度下降法 (Steepest Gradient Descent) 訓練模型 [cite: 10]
        """
        print("Running Gradient Descent...")
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化權重

        for i in range(self.max_iter):
            z = X @ self.w
            y_pred = self._sigmoid(z)
            # cross entropy
            # loss = - (1/n) * sum(y * log(y_pred) + (1 - y) * log(1 - y_pred))
            # gradient = (1/n) * X.T @ (y_pred - y)
            gradient = (1 / n_samples) * (X.T @ (y_pred - y))
            prev_w = self.w.copy()
            self.w = self.w - self.learning_rate * gradient
            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                print(f"Converged after {i+1} iterations.")
                break

        print(f"Gradient Descent Final Weights (w_0, w_1, w_2):\n{self.w}")
        return self.w

    def fit_newtons_method(self, X, y):
        print("\nRunning Newton's Method...")
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化權重

        for i in range(self.max_iter):
            y_pred = self._sigmoid(X @ self.w)
            gradient = (1 / n_samples) * (X.T @ (y_pred - y))
            S = np.diag(y_pred * (1 - y_pred))
            hessian = (1 / n_samples) * (X.T @ S @ X)

            prev_w = self.w.copy()
            if np.linalg.det(hessian) == 0 or np.linalg.cond(hessian) > 1e12:
                self.w = self.w - self.learning_rate * gradient
            else:
                self.w = self.w - np.linalg.inv(hessian) @ gradient

            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                print(f"Converged after {i+1} iterations.")
                break

        print(f"Newton's Method Final Weights (w_0, w_1, w_2):\n{self.w}")
        return self.w

    def predict(self, X, w):
        """
        使用訓練好的權重 w 進行預測
        """
        z = X @ w
        probs = self._sigmoid(z)
        return (probs >= 0.5).astype(int)  # 以 0.5 為閾值

    def calculate_metrics(self, y_true, y_pred):
        print("\nCalculating Metrics...")

        tn, fp, fn, tp = (0, 0, 0, 0)
        confusion_matrix = np.zeros((2, 2))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        sensitivity = 0.0
        if (tn + fp) > 0:
            sensitivity = tn / (tn + fp)
        else:
            sensitivity = 0.0

        specificity = 0.0
        if (tp + fn) > 0:
            specificity = tp / (tp + fn)
        else:
            specificity = 0.0

        print(f"Confusion Matrix:")
        print(f"          Predict cluster 1 | Predict cluster 2")
        print(f"Is cluster 1 | {tn:^15} | {fp:^15}")
        print(f"Is cluster 2 | {fn:^15} | {tp:^15}")
        print(f"\nSensitivity (Successfully predict cluster 1): {sensitivity:.5f}")
        print(f"Specificity (Successfully predict cluster 2): {specificity:.5f}")

        return confusion_matrix, sensitivity, specificity


def generate_data(
    n,
    mx: float,
    vx: float,
    my: float,
    vy: float,
):
    results = []
    cnt = 0

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)

    while cnt < n:
        u1 = np.random.uniform(1e-8, 1)
        u2 = np.random.uniform(1e-8, 1)
        z0_x = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1_x = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        u3 = np.random.uniform(1e-8, 1)
        u4 = np.random.uniform(1e-8, 1)
        z0_y = np.sqrt(-2 * np.log(u3)) * np.cos(2 * np.pi * u4)
        z1_y = np.sqrt(-2 * np.log(u3)) * np.sin(2 * np.pi * u4)

        x0 = mx + z0_x * sx
        y0 = my + z0_y * sy

        x1 = mx + z1_x * sx
        y1 = my + z1_y * sy

        results.append([x0, y0])
        results.append([x1, y1])

        cnt += 2

    return np.array(results)[:n]


def plot_results(D1, D2, w_gd, w_newton):
    print("\nPlotting results...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # 找出所有資料的邊界
    all_data = np.vstack((D1, D2))
    x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
    y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1

    # --- 1. Ground Truth  ---
    ax1.scatter(D1[:, 0], D1[:, 1], c="blue", marker="o", label="Cluster 1 (D1)")
    ax1.scatter(D2[:, 0], D2[:, 1], c="red", marker="x", label="Cluster 2 (D2)")
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # 繪製決策邊界
    def plot_boundary(ax, w, title, color, label):
        ax.scatter(D1[:, 0], D1[:, 1], c="blue", marker="o", label="Cluster 1 (D1)")
        ax.scatter(D2[:, 0], D2[:, 1], c="red", marker="x", label="Cluster 2 (D2)")

        if w is not None and len(w) == 3:
            plot_x = np.array([x_min, x_max])
            # 決策邊界: w_0 + w_1*x + w_2*y = 0
            # => y = (-w_0 - w_1*x) / w_2
            if w[2] != 0:  # 避免除以零 (垂直線)
                plot_y = (-w[0] - w[1] * plot_x) / w[2]
                ax.plot(plot_x, plot_y, c=color, lw=2, label=label)
            else:  # 畫一條垂直線
                v_line_x = -w[0] / w[1]
                ax.axvline(x=v_line_x, c=color, lw=2, label=label)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --- 2. Gradient Descent  ---
    plot_boundary(ax2, w_gd, "Gradient Descent", "green", "GD Boundary")

    # --- 3. Newton's Method  ---
    plot_boundary(ax3, w_newton, "Newton's Method", "purple", "Newton's Boundary")

    plt.tight_layout()
    plt.show()


def parse_arguments():
    """
    解析命令行參數
    """
    parser = argparse.ArgumentParser(description="Logistic Regression for ML HW4 [cite: 1, 2]")

    # 讀取 n [cite: 4]
    parser.add_argument("--n", type=int, default=50, help="Number of data points per cluster [cite: 4]")

    # 讀取 D1 和 D2 的高斯分佈參數 [cite: 5]
    # 使用 Case 1  作為預設值
    parser.add_argument("--mx1", type=float, default=1.0, help="Mean of x for D1")
    parser.add_argument("--vx1", type=float, default=2.0, help="Variance of x for D1")
    parser.add_argument("--my1", type=float, default=1.0, help="Mean of y for D1")
    parser.add_argument("--vy1", type=float, default=2.0, help="Variance of y for D1")

    parser.add_argument("--mx2", type=float, default=10.0, help="Mean of x for D2")
    parser.add_argument("--vx2", type=float, default=2.0, help="Variance of x for D2")
    parser.add_argument("--my2", type=float, default=10.0, help="Mean of y for D2")
    parser.add_argument("--vy2", type=float, default=2.0, help="Variance of y for D2")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 設定隨機亂數種子以便重現
    np.random.seed(42)

    # 1. 生成資料
    D1 = generate_data(args.n, args.mx1, args.vx1, args.my1, args.vy1)
    D2 = generate_data(args.n, args.mx2, args.vx2, args.my2, args.vy2)
    X = np.vstack((D1, D2))
    # 在 X 前加上一欄全為 1 的欄位以對應 w_0 (bias term)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_true = np.array([0] * args.n + [1] * args.n)  # D1 標記為 0, D2 標記為 1

    # 2. 梯度下降法
    model_gd = LogisticRegression()
    w_gd = model_gd.fit_gradient_descent(X, y_true)
    y_pred_gd = model_gd.predict(X, w_gd)
    model_gd.calculate_metrics(y_true, y_pred_gd)  #

    # 3. 牛頓法
    model_newton = LogisticRegression()
    w_newton = model_newton.fit_newtons_method(X, y_true)
    y_pred_newton = model_newton.predict(X, w_newton)
    model_newton.calculate_metrics(y_true, y_pred_newton)  #

    # 4. 視覺化
    plot_results(D1, D2, w_gd, w_newton)  # [cite: 15]


if __name__ == "__main__":
    main()
