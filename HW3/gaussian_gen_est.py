#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse


def univariate_gaussian_data_generator(m: float, s: float) -> float:
    """
    m: mean
    s: variance
    box-muller: https://blog.csdn.net/weixin_40920183/article/details/118716226
    """
    u1 = np.random.uniform(1e-8, 1)
    u2 = np.random.uniform(1e-8, 1)

    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    # z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    x0 = m + z0 * np.sqrt(s)
    # x1 = m + z1 * np.sqrt(s)

    return x0


class SequentialEstimator:

    def __init__(self, true_m: float, true_s: float):
        self.true_m = true_m
        self.true_s = true_s
        self.n_samples = 0
        self.estimated_mean = 0.0
        self.estimated_variance = 0.0
        self.m2 = 0.0
        self.threshold = 1e-3

    def run(self, max_iterations: int = 20):
        """
        welford algorithm: https://zhuanlan.zhihu.com/p/408474710
        """
        print(f"Data point source function: N({self.true_m}, {self.true_s})")

        for i in range(1, max_iterations + 1):
            point = univariate_gaussian_data_generator(self.true_m, self.true_s)
            self.n_samples += 1
            delta1 = point - self.estimated_mean  # x_n - mean_{n-1}
            self.estimated_mean += delta1 / self.n_samples  # mean_n
            delta2 = point - self.estimated_mean  # x_n - mean_n
            self.m2 += delta1 * delta2  # M2_n
            self.estimated_variance = self.m2 / (self.n_samples - 1)

            print(f"Add data point: {point:.5f}")
            print(f"Mean = {self.estimated_mean:.5f}  Variance = {self.estimated_variance:.5f}\n")
            if (
                abs(self.estimated_mean - self.true_m) < self.threshold
                and abs(self.estimated_variance - self.true_s) < self.threshold
            ):
                print("Converged!")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Parameter Estimation using Sequential Method")
    parser.add_argument("mean", help="mean")
    parser.add_argument("variance", help="variance")
    args = parser.parse_args()

    TRUE_MEAN = float(args.mean)
    TRUE_VARIANCE = float(args.variance)

    estimator = SequentialEstimator(true_m=TRUE_MEAN, true_s=TRUE_VARIANCE)

    estimator.run(max_iterations=1000000)
