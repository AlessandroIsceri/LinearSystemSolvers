import numpy as np
from solver import solver

class gradient_solver(solver):
    def __init__(self, A, b, tol, t):
        super().__init__(A, b, tol, t)

    def solve_dense(self):
        n = len(self.b)
        
        # Initialize variables: initial guess x and residual r
        x = np.zeros(n)
        r = np.zeros(n)
        k = 0

        # Iterate until convergence or maximum iterations
        while k < self.t and (np.linalg.norm(np.subtract(np.dot(self.A, x), self.b)) / np.linalg.norm(self.b)) >= self.tol:
            
            # Compute residual, alpha and new solution
            r = np.subtract(self.b, np.dot(self.A, x))
            alpha = np.dot(r, r) / np.dot(r, np.dot(self.A, r))
            x = np.add(x, alpha * r)
            
            k = k + 1
        
        return x, k
 
    def solve_sparse(self):
        n = len(self.b)
        
        # Initialize variables: initial guess x and residual r
        x = np.zeros(n)
        r = np.zeros(n)
        k = 0
                  
        # Iterate until convergence or maximum iterations  
        while k < self.t and (np.linalg.norm(np.subtract((self.A @ x), self.b)) / np.linalg.norm(self.b)) >= self.tol:
            
            # Compute residual, alpha and new solution
            r = np.subtract(self.b, (self.A @ x))
            alpha = np.dot(r, r) / np.dot(r, (self.A @ r))
            x = np.add(x, alpha * r)
            
            k = k + 1

        return x, k