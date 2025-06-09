import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from conjugated_gradient_solver import conjugated_gradient_solver
from gauss_seidel_solver import gauss_seidel_solver
from gradient_solver import gradient_solver
from jacobi_solver import jacobi_solver
from utils import compute_relative_error

def main(A, b, x, tol):
    """
    Execute four iterative solvers (Jacobi, Gauss-Seidel, Gradient, Conjugate Gradient)
    on system Ax = b with known exact solution x and given tolerance tol.

    Parameters:
    - A: coefficient matrix (dense or sparse)
    - b: right-hand side vector
    - x: exact solution vector for error computation
    - tol: convergence tolerance

    Returns:
    - relative_errors: list of relative errors for each solver
    - elapsed_times: list of elapsed execution times for each solver
    - it_numbers: list of iteration counts for each solver
    """

    t = 20000
    # Initialize solvers with the same system, tolerance, and max iterations
    solvers = [jacobi_solver(A, b, tol, t), 
               gauss_seidel_solver(A, b, tol, t), 
               gradient_solver(A, b, tol, t), 
               conjugated_gradient_solver(A, b, tol, t)]
    
    relative_errors = [0 for i in range(4)]
    it_numbers = [0 for i in range(4)]
    elapsed_times = [0 for i in range(4)]
    i = 0

    for cur_solver in solvers:
        print("solver: ", type(cur_solver).__name__)
        start = time.time()
        
        # Solve the system using the current solver and compute relative error
        x_approx, it_number = cur_solver.solve()
        relative_error = compute_relative_error(x_approx, x)
        
        end = time.time()
        elapsed_time = end - start

        relative_errors[i] = relative_error
        it_numbers[i] = it_number
        elapsed_times[i] = elapsed_time
        
        i = i + 1

    return relative_errors, elapsed_times, it_numbers