from solver import solver
import numpy as np

class conjugated_gradient_solver(solver):
 
    def __init__(self, A, b, tol, t):
        super().__init__(A, b, tol, t)

    def solve_dense(self):
        n = len(self.b)

        # Initialize variables: initial guess x, residual r, and search direction d
        x = np.zeros(n)
        r = np.subtract(self.b, np.dot(self.A, x))
        d = r.copy()

        k = 0
        
        # Iterate until convergence or maximum iterations
        while k < self.t and (np.linalg.norm(np.subtract(np.dot(self.A, x), self.b)) / np.linalg.norm(self.b)) >= self.tol:
            y = np.dot(self.A, d)

            # Compute step size alpha
            alpha = np.dot(d, r) / np.dot(d, y)
            x = np.add(x, alpha * d)
            
            # Update residual
            r = np.subtract(self.b, np.dot(self.A, x))
            w = np.dot(self.A, r)
            
            # Compute beta for new direction
            beta = np.dot(d, w) / np.dot(d, y)
            d = np.subtract(r, beta * d)
            
            k = k + 1
        
        return x, k

    def solve_sparse(self):
        n = len(self.b)

        x = np.zeros(n)
        r = np.subtract(self.b, (self.A @ x))
        d = r.copy()

        k = 0

        # Iterate until convergence or maximum iterations
        while k < self.t and (np.linalg.norm(np.subtract((self.A @ x), self.b)) / np.linalg.norm(self.b)) >= self.tol:

            y = self.A @ d
            
            # Compute step size alpha
            alpha = np.dot(d, r) / np.dot(d, y)
            x = np.add(x, alpha * d)
            
            # Update residual
            r = np.subtract(self.b, (self.A @ x))
            w = self.A @ r
            
            # Compute beta for new direction
            beta = np.dot(d, w) / np.dot(d, y)
            d = np.subtract(r, beta * d)
            
            k = k + 1

        return x, k