import numpy as np
from abc import abstractmethod, ABC
from scipy.sparse import issparse

from utils import is_spd, is_spd_sparse

# Abstract base class for linear system solvers
class solver(ABC):

    def __init__(self, A, b, tol, t):
        """
        Initializes the solver instance.

        Parameters:
        - A: The coefficient matrix (can be dense or sparse)
        - b: The right-hand side vector
        - tol: Convergence tolerance
        - t: Number of iteration limit
        """
        
        self.A = A
        self.b = b
        self.tol = tol
        self.t = t
        self.check_matrix()

    def check_matrix(self):
      """
      Validates the input matrix and vector:
      - Ensures A is square
      - Checks that b has a matching number of rows
      - Verifies that A is symmetric positive definite (SPD)
      """
      self.isMatrixValid = True

      if issparse(self.A):
        # Validation for sparse matrix
        if self.A.shape[0] != self.A.shape[1]:
            self.isMatrixValid = False
        if len(self.b) != self.A.shape[0]:
            self.isMatrixValid = False
        if not(is_spd_sparse(self.A)):
            self.isMatrixValid = False
      else:
        # Validation for dense matrix
        if len(self.A) != len(self.A[0]):
            self.isMatrixValid = False
        if len(self.b) != len(self.A):
            self.isMatrixValid = False
        if not(is_spd(self.A)):
            self.isMatrixValid = False


    def solve(self):
        """
        Main interface to solve the linear system Ax = b.

        Returns:
        - A NumPy array with the solution
        - The number of iterations used, or -1 if the matrix is invalid
        """
        
        n = len(self.b)
        
        if self.isMatrixValid == False:
          return np.zeros(n), -1

        if issparse(self.A):
            return self.solve_sparse()
        else:
            return self.solve_dense()


    @abstractmethod
    def solve_dense(self):
        """
        Abstract method to solve the system when A is a dense matrix.
        Must be implemented in derived classes.
        """
        pass

    @abstractmethod
    def solve_sparse(self):
        """
        Abstract method to solve the system when A is a sparse matrix.
        Must be implemented in derived classes.
        """
        pass