from solver import solver
import numpy as np
from scipy.sparse import diags

class jacobi_solver(solver):

    def __init__(self, A, b, tol, t):
        super().__init__(A, b, tol, t)

    def solve_dense(self):
        n = len(self.b)

        # Matrix splitting: L + U = A - D
        D = np.diag(np.diag(self.A))
        inverse_D = np.diag(1 / np.diag(D))
        L_U = np.subtract(self.A, D)

        # Initialize solution vector
        x = np.zeros(n)
        k = 0

        # Iterate until convergence or maximum iterations
        while (np.linalg.norm(np.subtract(np.dot(self.A, x), self.b)) / np.linalg.norm(self.b)) >= self.tol and k < self.t:
            x = np.dot(inverse_D, np.subtract(self.b, np.dot(L_U, x)))
            
            k = k + 1

        return x, k 

    def solve_sparse(self):
        n = len(self.b)

        # Matrix splitting: L + U = A - D
        D = diags(self.A.diagonal())
        inverse_D = diags(1 / (D.diagonal()))
        L_U = self.A - D

        # Initialize solution vector
        x = np.zeros(n)
        k = 0

        # Iterate until convergence or maximum iterations
        while (np.linalg.norm(np.subtract((self.A @ x), self.b)) / np.linalg.norm(self.b)) >= self.tol and k < self.t:
            x = inverse_D @ np.subtract(self.b, (L_U @ x))
            k = k + 1

        return x, k