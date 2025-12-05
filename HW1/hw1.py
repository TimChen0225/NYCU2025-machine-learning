import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# --- Core Matrix Operations (Self-implemented as required) ---

def lu_decomposition(matrix):
    """
    see https://www.geeksforgeeks.org/dsa/doolittle-algorithm-lu-decomposition/
      L   *   U   =   A
    1 0 0   d e f   j k l
    a 1 0   0 g h   m n o
    b c 1   0 0 i   p q r
    r = bf + ch + i, so i = r - bf - ch => U[2][2] = A[2][2] - L[2][0]*U[0][2] - L[2][1]*U[1][2]
    q = be + cg, so c = (q - be) / g => L[2][1] = (A[2][1] - L[2][0]*U[0][1]) / U[1][1]
    """
    n = matrix.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_val = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - sum_val
        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_val = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (matrix[k][i] - sum_val) / U[i][i]
    return L, U

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # Handle potential division by zero
        if U[i, i] == 0:
            raise np.linalg.LinAlgError("Singular matrix: Zero pivot encountered.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def matrix_inverse(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    n = matrix.shape[0]
    identity = np.identity(n)
    inv = np.zeros((n, n))
    
    try:
        L, U = lu_decomposition(matrix)
        for i in range(n):
            # For each column of the identity matrix, solve Ax = b
            # which is equivalent to LUx = b.
            # 1. Solve Ly = b using forward substitution
            y = forward_substitution(L, identity[:, i])
            # 2. Solve Ux = y using backward substitution
            inv[:, i] = backward_substitution(U, y)
    except np.linalg.LinAlgError as e:
        print(f"Error during matrix inversion: {e}", file=sys.stderr)
        print("Cannot compute the closed-form solution for LSE or Newton's Method.", file=sys.stderr)
        return None
        
    return inv

# --- Regression Algorithms ---

def lse_closed_form(A, y, lambda_reg):
    print("LSE:")
    n_bases = A.shape[1]
    AtA = A.T @ A

    # L2 norm
    regularized_AtA = AtA + lambda_reg * np.identity(n_bases)
    inv_matrix = matrix_inverse(regularized_AtA)
    
    if inv_matrix is None:
        return None

    Aty = A.T @ y
    weights = inv_matrix @ Aty
    return weights

def steepest_descent(A, y, lambda_reg, learning_rate=0.0001, iterations=100000):
    print("Steepest Descent:")
    n_samples, n_bases = A.shape
    weights = np.zeros(n_bases) 

    for i in range(iterations):
        # Prediction
        y_pred = A @ weights
        
        # Error term for LSE part
        error_term = y_pred - y
        
        # Gradient of the LSE part: A^T * (A*w - y)
        grad_lse = A.T @ error_term
        
        # Subgradient of the L1 regularization part: lambda * sign(w)
        grad_l1 = lambda_reg * np.sign(weights)
        
        # Total gradient
        gradient = grad_lse + grad_l1
        
        # Update weights
        weights = weights - learning_rate * gradient

        # Optional: Print progress for long runs
        # if i % 10000 == 0:
        #     print(f"Iteration {i}, Error: {calculate_error(A, y, weights)}")
            
    return weights

def newtons_method(A, y):
    print("Newton's Method:")
    
    # Hessian H = A^T * A
    hessian = A.T @ A
    
    # Inverse of Hessian using the self-implemented function
    inv_hessian = matrix_inverse(hessian)

    if inv_hessian is None:
        return None

    # Gradient = A^T * A * w - A^T * y. At w=0, Gradient = -A^T*y
    # Or more directly, the solution is w = H^-1 * A^T*y
    Aty = A.T @ y
    weights = inv_hessian @ Aty
    
    return weights

# --- Helper Functions ---

def load_data(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except Exception as e:
        print(f"Error loading data file: {e}", file=sys.stderr)
        sys.exit(1)

def build_design_matrix(x, n_bases):
    A = np.zeros((len(x), n_bases))
    for i in range(n_bases):
        A[:, i] = x ** i
    return A

def calculate_error(A, y, weights):
    if weights is None:
        return float('inf')
    predictions = A @ weights
    error = np.sum((y - predictions) ** 2)
    return error

def format_equation(weights):
    if weights is None:
        return "Fitting failed."
    equation_parts = []
    for i, w in enumerate(reversed(weights)):
        power = len(weights) - 1 - i
        if abs(w) < 1e-6:  # Skip terms with very small coefficients
            continue
        
        sign = "-" if w < 0 else "+"
        
        # Hide '+' for the first term
        if not equation_parts:
            sign = "-" if w < 0 else ""

        w_abs = abs(w)
        
        if power == 0:
            equation_parts.append(f"{sign} {w_abs:.10f}")
        elif power == 1:
            equation_parts.append(f"{sign} {w_abs:.10f} x")
        else:
            equation_parts.append(f"{sign} {w_abs:.10f} x^{power}")
            
    return " ".join(equation_parts).lstrip('+ ')

def visualize(x, y, weights, n_bases, title):
    if weights is None:
        print(f"Skipping visualization for '{title}' due to fitting failure.")
        return
        
    plt.figure(figsize=(10, 6))
    # Plot original data points
    plt.scatter(x, y, color='red', label='Data Points')
    
    # Generate points for the curve
    x_curve = np.linspace(min(x), max(x), 200)
    A_curve = build_design_matrix(x_curve, n_bases)
    y_curve = A_curve @ weights
    
    # Plot the fitting curve
    plt.plot(x_curve, y_curve, color='black', label='Fitting Curve')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Perform polynomial linear regression.")
    parser.add_argument("filepath", type=str, help="Path to the data file (e.g., 'data.txt')")
    parser.add_argument("n_bases", type=int, help="The number of polynomial bases (n)")
    parser.add_argument("lambda_reg", type=float, help="Lambda for regularization")
    args = parser.parse_args()

    # Unpack arguments
    filepath = args.filepath
    n_bases = args.n_bases
    lambda_reg = args.lambda_reg

    # Load data and build design matrix
    x, y = load_data(filepath)
    A = build_design_matrix(x, n_bases)

    # --- Run and report for each method ---
    print("Data loaded successfully.")

    # 1. Closed-form LSE
    weights_lse = lse_closed_form(A, y, lambda_reg)
    if weights_lse is not None:
        error_lse = calculate_error(A, y, weights_lse)
        print(f"Fitting line: {format_equation(weights_lse)}")
        print(f"Total error: {error_lse}\n")

    # 2. Steepest Descent
    weights_sd = steepest_descent(A, y, lambda_reg)
    if weights_sd is not None:
        error_sd = calculate_error(A, y, weights_sd)
        print(f"Fitting line: {format_equation(weights_sd)}")
        print(f"Total error: {error_sd}\n")

    # 3. Newton's Method
    weights_newton = newtons_method(A, y)
    if weights_newton is not None:
        error_newton = calculate_error(A, y, weights_newton)
        print(f"Fitting line: {format_equation(weights_newton)}")
        print(f"Total error: {error_newton}\n")

    # --- Visualization ---
    visualize(x, y, weights_lse, n_bases, f"LSE (n={n_bases}, lambda={lambda_reg})")
    visualize(x, y, weights_sd, n_bases, f"Steepest Descent (n={n_bases}, lambda={lambda_reg})")
    visualize(x, y, weights_newton, n_bases, f"Newton's Method (n={n_bases})")


if __name__ == "__main__":
    main()