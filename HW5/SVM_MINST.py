import numpy as np
import argparse
import time
from libsvm.svmutil import *
from scipy.spatial.distance import cdist
import itertools


def load_svm_data(x_file, y_file):
    try:
        X_data = np.loadtxt(x_file, delimiter=",")
        Y_data = np.loadtxt(y_file, delimiter=",")

        Y_list = Y_data.ravel().tolist()
        X_list = X_data.tolist()

        return Y_list, X_list

    except Exception as e:
        print(f"loading data failed: {e}")
        return None, None


def task1_compare_kernels(train_problem, y_test, x_test):
    print("\n--- task 1: compare different SVM kernels ---\n")

    kernel_options = {
        "Linear": "-t 0",
        "Polynomial": "-t 1",
        "RBF": "-t 2",
    }

    result = {}

    for kernel_name, kernel_param_str in kernel_options.items():
        print(f"training and testing with {kernel_name} kernel...")
        param = svm_parameter(kernel_param_str + " -c 1 -q")
        model = svm_train(train_problem, param)
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
        result[kernel_name] = p_acc[0]
        print("")

    return result


def task2_grid_search(train_problem, y_test, x_test):
    print("\n--- Task 2: Grid Search ---\n")
    C_range = 2.0 ** np.arange(-5, 16, 2)
    Gamma_range = 2.0 ** np.arange(-15, 4, 2)
    Degree_range = [2, 3, 4]

    kernel_options = ["Linear", "Polynomial", "RBF"]

    best_results = {}

    for kernel_name in kernel_options:
        print(f"--- Grid Search for {kernel_name} Kernel ---")
        best_acc = -1
        best_params = {}
        if kernel_name == "Linear":
            for C in C_range:
                param_str = f"-t 0 -c {C} -q -v 5"
                print(f"Training Linear SVM with C={C}...")
                param = svm_parameter(param_str)
                acc = svm_train(train_problem, param)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C}

        elif kernel_name == "Polynomial":
            for C, degree in itertools.product(C_range, Degree_range):
                param_str = f"-t 1 -c {C} -d {degree} -q -v 5"
                print(f"Training Polynomial SVM with C={C}, degree={degree}...")
                param = svm_parameter(param_str)
                acc = svm_train(train_problem, param)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "degree": degree}

        elif kernel_name == "RBF":
            for C, gamma in itertools.product(C_range, Gamma_range):
                param_str = f"-t 2 -c {C} -g {gamma} -q -v 5"
                print(f"Training RBF SVM with C={C}, gamma={gamma}...")
                param = svm_parameter(param_str)
                acc = svm_train(train_problem, param)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "gamma": gamma}

        best_results[kernel_name] = {"best_cv_acc": best_acc, "best_params": best_params}
        print(f"Best {kernel_name} Kernel Accuracy: {best_acc:.4f} with params: {best_params}\n")

    for kernel_name, result in best_results.items():
        param = result["best_params"]
        param_str = ""
        if kernel_name == "Linear":
            param_str = f"-t 0 -c {param['C']} -q"
        elif kernel_name == "Polynomial":
            param_str = f"-t 1 -c {param['C']} -d {param['degree']} -q"
        elif kernel_name == "RBF":
            param_str = f"-t 2 -c {param['C']} -g {param['gamma']} -q"
        plablel, p_acc, p_val = svm_predict(y_test, x_test, svm_train(train_problem, svm_parameter(param_str)))
        best_results[kernel_name]["test_acc"] = p_acc[0]

    return best_results


def task3_combined_kernel(y_train, x_train, y_test, x_test):
    print("\n--- task 3: custom combined kernel ---\n")
    C_range = 2.0 ** np.arange(-5, 16, 2)
    Gamma_range = 2.0 ** np.arange(-15, 4, 2)
    Weight_range = np.arange(0.2, 0.8, 0.2)

    best_acc = -1
    best_params = {}

    ids = np.arange(1, len(y_train) + 1).reshape(-1, 1)

    for g, w in itertools.product(Gamma_range, Weight_range):
        K_matrix = custom_kernel(x_train, x_train, gamma=g, weight=w)
        K_with_ids = np.hstack((ids, K_matrix)).tolist()
        prob = svm_problem(y_train, K_with_ids, isKernel=True)
        for C in C_range:
            param = svm_parameter(f"-t 4 -c {C} -v 5 -q")
            acc = svm_train(prob, param)
            if acc > best_acc:
                best_acc = acc
                best_params = {"C": C, "gamma": g, "weight": w}
    print(f"Best Combined Kernel Accuracy: {best_acc:.2f} with params: {best_params}\n")
    final_matrix = custom_kernel(x_train, x_train, gamma=best_params["gamma"], weight=best_params["weight"])
    final_matrix_with_ids = np.hstack((ids, final_matrix)).tolist()
    prob = svm_problem(y_train, final_matrix_with_ids, isKernel=True)
    model = svm_train(prob, svm_parameter(f"-t 4 -c {best_params['C']} -q"))

    test_matrix = custom_kernel(x_test, x_train, gamma=best_params["gamma"], weight=best_params["weight"])
    test_ids = np.arange(1, len(y_test) + 1).reshape(-1, 1)
    test_matrix_with_ids = np.hstack((test_ids, test_matrix)).tolist()
    p_label, p_acc, p_val = svm_predict(y_test, test_matrix_with_ids, model, "-q")
    print(f"Test Accuracy with Combined Kernel: {p_acc[0]:.2f}")

    best_result = {"best_cv_acc": best_acc, "best_params": best_params, "test_acc": p_acc[0]}
    return best_result


def custom_kernel(x1, x2, gamma, weight):
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    # 1. Linear Part
    K_lin = np.dot(x1_arr, x2_arr.T)

    # 2. RBF Part
    dists_sq = cdist(x1_arr, x2_arr, metric="sqeuclidean")
    K_rbf = np.exp(-gamma * dists_sq)

    K_combined = (weight * K_lin) + ((1 - weight) * K_rbf)

    return K_combined


def main():
    parser = argparse.ArgumentParser(description="SVM on MNIST for ML HW5")
    parser.add_argument("--x_train", type=str, default="./ML_HW05/data/X_train.csv", help="Path to X_train.csv")
    parser.add_argument("--y_train", type=str, default="./ML_HW05/data/Y_train.csv", help="Path to Y_train.csv")
    parser.add_argument("--x_test", type=str, default="./ML_HW05/data/X_test.csv", help="Path to X_test.csv")
    parser.add_argument("--y_test", type=str, default="./ML_HW05/data/Y_test.csv", help="Path to Y_test.csv")

    args = parser.parse_args()
    print(f"args: {args}")

    y_train, x_train = load_svm_data(args.x_train, args.y_train)
    y_test, x_test = load_svm_data(args.x_test, args.y_test)

    if y_train is None or x_train is None or y_test is None or x_test is None:
        print("data loading failed, exiting.")
        return

    train_problem = svm_problem(y_train, x_train)

    task1_compare_kernels(train_problem, y_test, x_test)

    task2_grid_search(train_problem, y_test, x_test)

    task3_combined_kernel(y_train, x_train, y_test, x_test)

    print("\nSVM finished.")


if __name__ == "__main__":
    main()
