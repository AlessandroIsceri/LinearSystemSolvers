import numpy as np

def compute_relative_error(x_approx, x):
    return np.linalg.norm(np.subtract(x, x_approx)) / np.linalg.norm(x)

def is_spd(x):
    n = len(x)
    eps = 10**(-9)
    for i in range(n):
        for j in range(i):
            if (x[i][j] - x[j][i] > eps):
                return False
    return np.all(np.linalg.eigvals(x) > 0)

def is_spd_sparse(x):
    x = x.toarray()
    return is_spd(x)

class solve_lower:
    def __init__(self, L, b):
        self.L = L
        self.b = b
        self.isMatrixValid = True
        
        n = len(self.b)
        #check if det(L) != 0
        for i in range(n):
            if abs(self.L[i][i]) < 10 ** (-15):
                self.isMatrixValid = False
                break

    def set_b(self, b):
        self.b = b

    def solve(self):
        if self.isMatrixValid == False:
            return False
        
        n = len(self.b)
        x = np.zeros(n)

        x[0] = self.b[0] / self.L[0][0]
        for i in range(1, n):
            x[i] = (self.b[i]-(np.dot(self.L[i, 0:i], x[0:i]))) / self.L[i][i]

        return x