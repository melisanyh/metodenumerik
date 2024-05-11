import metodebalikan as np
import unittest

def inverse_matrix_method(matrix_A, matrix_B):
    # Pastikan matriks_A adalah matriks persegi
    if matrix_A.shape[0] != matrix_A.shape[1]:
        raise ValueError("Matriks koefisien harus berbentuk persegi.")
    
    # Pastikan matriks_B memiliki jumlah baris yang sama dengan matriks_A
    if matrix_A.shape[0] != matrix_B.shape[0]:
        raise ValueError("Jumlah baris matriks koefisien harus sama dengan jumlah baris vektor hasil.")
    
    # Cari matriks balikan dari matriks_A
    A_inv = np.linalg.inv(matrix_A)
    
    # Hitung solusi menggunakan matriks balikan
    solution = np.dot(A_inv, matrix_B)
    
    return solution

class TestInverseMatrixMethod(unittest.TestCase):
    def test_inverse_matrix_method(self):
        # Persiapan data pengujian
        A = np.array([[2, 1], [1, -1]])
        B = np.array([5, -2])
        
        # Hasil yang diharapkan
        expected_solution = np.array([2, 1])
        
        # Panggil fungsi yang ingin diuji
        actual_solution = inverse_matrix_method(A, B)
        
        # Lakukan pengecekan
        np.testing.assert_array_almost_equal(actual_solution, expected_solution)

if __name__ == '__main__':
    unittest.main()