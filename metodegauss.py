import metodegauss as np
import unittest

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.copy(A)

    for i in range(n):
        # Pivoting
        max_index = np.argmax(np.abs(U[i:, i])) + i
        if max_index != i:
            U[[i, max_index]] = U[[max_index, i]]
            L[[i, max_index]] = L[[max_index, i]]
        
        pivot = U[i, i]
        if pivot == 0:
            raise ValueError("Matrix is singular.")
        
        # Elimination
        for j in range(i+1, n):
            factor = U[j, i] / pivot
            L[j, i] = factor
            U[j] -= factor * U[i]

    return L, U

class TestLUDecomposition(unittest.TestCase):
    def test_lu_decomposition(self):
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        L, U = lu_decomposition(A)
        np.testing.assert_allclose(np.dot(L, U), A)

if __name__ == "__main__":
    unittest.main()