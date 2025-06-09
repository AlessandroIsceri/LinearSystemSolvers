# LinearSystemSolvers

**LinearSystemSolvers** is a Python library that provides implementations of four iterative methods to solve linear systems of the form **Ax = b**.

## Implemented methods:

- Jacobi Method  
- Gauss-Seidel Method  
- Gradient Method  
- Conjugate Gradient Method  

These solvers can work for both dense and sparse systems, and can be used for educational purposes, performance comparisons, or integration into larger numerical projects.

## Solver Output

Each solver class (`jacobi_solver`, `gauss_seidel_solver`, `gradient_solver`, `conjugated_gradient_solver`) implements a `.solve()` method that returns x_approx and it_number.
If the input matrix A or the vector b is not valid the method returns -1 as it_number.


## Technologies Used

- Python 3
- NumPy
- SciPy

---

## Project Structure

<pre>
LinearSystemSolvers/
├── data/                     # Matrices in .mtx format
├── examples/                 # Usage examples
├── images/                   # Output images for reports
└── src/                      # Library source code
    ├── conjugated_gradient_solver.py
    ├── gauss_seidel_solver.py
    ├── gradient_solver.py
    ├── jacobi_solver.py
    ├── solver.py
    └── utils.py
</pre>

---

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/AlessandroIsceri/LinearSystemSolvers.git
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. Import the solvers from the src/ folder.

```python
import numpy as np
from jacobi_solver import jacobi_solver


# Define the coefficient matrix A 
A = np.array([[4.0, -1.0, 0.0, 0.0],
              [-1.0, 4.0, -1.0, 0.0],
              [0.0, -1.0, 4.0, -1.0],
              [0.0, 0.0, -1.0, 3.0]])

# Define the right-hand side vector b
b = np.array([15.0, 10.0, 10.0, 10.0])

# Set tolerance and maximum number of iterations
tol = 1e-6
t = 1000

solver = jacobi_solver(A, b, tol, t)
x_approx, it_number = solver.solve()

# Print results
print("Approximate solution x: ", x_approx)
print("Number of iterations: ", it_number)
```

## More Information
For more information on the project and obtained results, please read the report: [`Relazione_Progetto_1_Metodi_Farioli_Isceri.pdf`](https://github.com/AlessandroIsceri/LinearSystemSolvers/blob/master/Relazione_Progetto_1_Metodi_Farioli_Isceri.pdf).