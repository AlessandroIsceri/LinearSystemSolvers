import numpy as np
from solver import solver
from scipy.sparse import diags, tril, triu

from utils import solve_lower

class gauss_seidel_solver(solver):
    def __init__(self, A, b, tol, t):
        super().__init__(A, b, tol, t)


    def solve_dense(self):
        n = len(self.b)
        
        # Matrix splitting: A = (D - L) - U
        D = np.diag(np.diag(self.A))
        L = np.add(np.tril(-self.A), D)
        U = np.add(np.triu(-self.A), D)
        B = np.subtract(D, L)
        
        # Initialize variables: initial guess x and lower triangular solver
        x = np.zeros(n)
        triangular_solver = solve_lower(B, np.zeros(n))
        
        k = 0

        # Iterate until convergence or maximum iterations
        while (np.linalg.norm(np.subtract(np.dot(self.A, x), self.b)) / np.linalg.norm(self.b)) >= self.tol and k < self.t:
            
            # Compute new right-hand side and solve the lower triangular system Bx = f
            f = np.add(np.dot(U, x), self.b)
            triangular_solver.set_b(f)
            x = triangular_solver.solve()

            k = k + 1
            
        return x, k


    def solve_sparse(self):
        n = len(self.b)
        
        # Matrix splitting: A = (D - L) - U
        D = diags(self.A.diagonal())
        L = tril(-self.A) + D
        U = triu(-self.A) + D
        B = D - L 
        B = np.array(B.todense())
        
        # Initialize variables: initial guess x and lower triangular solver
        x = np.zeros(n)
        triangular_solver = solve_lower(B, np.zeros(n))

        k = 0

        # Iterate until convergence or maximum iterations
        while (np.linalg.norm(np.subtract((self.A @ x), self.b)) / np.linalg.norm(self.b)) >= self.tol and k < self.t:

            # Compute new right-hand side and solve the lower triangular system Bx = f
            f = np.add((U @ x), self.b)
            triangular_solver.set_b(f)
            x = triangular_solver.solve()

            k = k + 1
            
        return x, k