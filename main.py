import numpy as np

ITERATION_LIMIT = 1000

matrixAB = np.loadtxt('matrix.txt')
A = np.copy(matrixAB[:, :matrixAB.shape[0]])
B = np.copy(matrixAB[:, matrixAB.shape[1] - 1])

""""Prints matrix"""
print("Matrix:")
print(matrixAB)
print()

""""Iteration loops"""
x = np.zeros_like(B)
for iter_count in range(ITERATION_LIMIT):
    print("X vector: ", x, "iteration: ", iter_count)
    x_new = np.zeros_like(x)

    for i in range(A.shape[0]):
        alpha1 = np.dot(A[i, :i], x[:i])
        alpha2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (B[i] - alpha1 - alpha2) / A[i, i]

    if np.allclose(x, x_new, atol=1e-10, rtol=0.):
        break

    x = x_new

""""Prints solutions"""
print()
print("Solution: ")
print("Vector X: ", x, "Total iterations: ", iter_count)

""""Error check"""
error = np.dot(A, x) - B
print("Error: ", error)
