import time
from conjugated_gradient_solver import conjugated_gradient_solver
from gauss_seidel_solver import gauss_seidel_solver
from gradient_solver import gradient_solver
from jacobi_solver import jacobi_solver
from utils import compute_relative_error

#execute the 4 implemented methods on matrix A, right-hand side vector b, with exact solution x and tolerance = tol
def main(A, b, x, tol):

    t = 20000
    solvers = [jacobi_solver(A, b, tol, t), gauss_seidel_solver(A, b, tol, t), gradient_solver(A, b, tol, t), conjugated_gradient_solver(A, b, tol, t)]
    relative_errors = [0 for i in range(4)]
    it_numbers = [0 for i in range(4)]
    elapsed_times = [0 for i in range(4)]
    i = 0

    for cur_solver in solvers:
        print("solver: ", type(cur_solver).__name__)
        start = time.time()
        x_approx, it_number = cur_solver.solve()
        relative_error = compute_relative_error(x_approx, x)
        end = time.time()
        elapsed_time = end - start

        relative_errors[i] = relative_error
        it_numbers[i] = it_number
        elapsed_times[i] = elapsed_time
        i = i + 1

    return relative_errors, elapsed_times, it_numbers