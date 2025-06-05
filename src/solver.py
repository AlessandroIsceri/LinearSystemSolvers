import numpy as np
from abc import abstractmethod, ABC
from scipy.sparse import issparse

from utils import is_spd, is_spd_sparse

class solver(ABC):

    def __init__(self, A, b, tol, t):
        self.A = A
        self.b = b
        self.tol = tol
        self.t = t
        self.check_matrix()

    #checks if the matrix is valid (same number of rows and cols, same number of rows of b, spd)
    def check_matrix(self):
      self.isMatrixValid = True

      if issparse(self.A):
        if self.A.shape[0] != self.A.shape[1]:
            self.isMatrixValid = False

        if len(self.b) != self.A.shape[0]:
            self.isMatrixValid = False

        if not(is_spd_sparse(self.A)):
            self.isMatrixValid = False
      else:
        if len(self.A) != len(self.A[0]):
            self.isMatrixValid = False

        if len(self.b) != len(self.A):
            self.isMatrixValid = False

        if not(is_spd(self.A)):
            self.isMatrixValid = False


    def solve(self):
        n = len(self.b)
        #if the matrix is not valid returns -1 as iteration number
        if self.isMatrixValid == False:
          return np.zeros(n), -1

        if issparse(self.A):
            return self.solve_sparse()
        else:
            return self.solve_dense()


    @abstractmethod
    def solve_dense(self):
        pass

    @abstractmethod
    def solve_sparse(self):
        pass