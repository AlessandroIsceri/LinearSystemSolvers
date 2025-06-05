import os
import sys
import numpy as np
from scipy.io import mmread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import main

relative_path = '../data/'
matrix_names = ['spa1.mtx', 'spa2.mtx', 'vem1.mtx', 'vem2.mtx']
tols = [10**(-4), 10**(-6), 10**(-8), 10**(-10)]
#matrice, tol, metodo
errors_sparse = np.zeros((4, 4, 4))
elapsed_times_sparse = np.zeros((4, 4, 4))
it_numbers_sparse = np.zeros((4, 4, 4))

i = 0
j = 0

for matrix_name in matrix_names:
    j = 0
    A = mmread(relative_path + matrix_name)
    n = A.shape[0]
    x_esatta = [1 for i in range(n)]
    b = A @ x_esatta
    print("matrix_name: ", matrix_name)
    for tol in tols:
        print("tol: ", tol)
        errors_sparse[i][j], elapsed_times_sparse[i][j], it_numbers_sparse[i][j] = main(A, b, x_esatta, tol)
        j = j + 1
    i = i + 1

print(errors_sparse)

print(it_numbers_sparse)

print(elapsed_times_sparse)