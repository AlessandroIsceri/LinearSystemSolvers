import os
import sys
import numpy as np
from scipy.io import mmread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import main

relative_path = '../data/'
matrix_names = ['spa1.mtx', 'spa2.mtx', 'vem1.mtx', 'vem2.mtx']
tols = [10**(-4), 10**(-6), 10**(-8), 10**(-10)]

errors = np.zeros((4, 4, 4))
elapsed_times = np.zeros((4, 4, 4))
it_numbers = np.zeros((4, 4, 4))

i = 0
j = 0

for matrix_name in matrix_names:
    j = 0
    A = mmread(relative_path + matrix_name)
    A = np.array(A.todense())
    n = len(A)
    x_esatta = [1 for i in range(n)]
    b = np.dot(A, x_esatta)
    print("matrix_name: ", matrix_name)
    for tol in tols:
        print("tol: ", tol)
        errors[i][j], elapsed_times[i][j], it_numbers[i][j] = main(A, b, x_esatta, tol)
        j = j + 1
    i = i + 1
    
    
print(errors)

print(it_numbers)

print(elapsed_times)