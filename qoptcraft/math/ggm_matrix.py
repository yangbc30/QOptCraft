import numpy as np
from typing import List, Tuple
import math

def generate_ggm_matrices(m: int, normalized: bool = True) -> List[np.ndarray]:
    """
    Generate m×m dimensional Generalized Gell-Mann Matrices (GGM)
    
    Parameters:
        m: Matrix dimension
        normalized: Whether to normalize (default True, normalized by 1/√2 factor)
    
    Returns:
        List containing all m²-1 GGM matrices
    """
    if m < 1:
        raise ValueError("dimension error")

    if m == 1:
        return [np.eye(1)]
    
    matrices = []
    
    # 1. Symmetric matrices: λ_k^(m) = 1/√2 (E_kj + E_jk), k < j
    for k in range(m):
        for j in range(k + 1, m):
            matrix = np.zeros((m, m), dtype=complex)
            matrix[k, j] = 1
            matrix[j, k] = 1
            if normalized:
                matrix = matrix / math.sqrt(2)
            matrices.append(matrix)
    
    # 2. Antisymmetric matrices: λ_i^(m) = -i/√2 (E_kj - E_jk), k > j
    for k in range(m):
        for j in range(k + 1, m):
            matrix = np.zeros((m, m), dtype=complex)
            matrix[k, j] = -1j
            matrix[j, k] = 1j
            if normalized:
                matrix = matrix / math.sqrt(2)
            matrices.append(matrix)
    
    # 3. Diagonal matrices: λ_l^(m) = 1/√(l(l+1)) (∑_{j=1}^l E_jj - l E_{l+1,l+1}), 1 ≤ l ≤ m-1
    for l in range(1, m):
        matrix = np.zeros((m, m), dtype=complex)
        # First l diagonal elements set to 1
        for j in range(l):
            matrix[j, j] = 1
        # The (l+1)-th diagonal element set to -l
        matrix[l, l] = -l
        
        # Normalization factor
        normalization_factor = 1 / math.sqrt(l * (l + 1))
        matrix = matrix * normalization_factor
        
        matrices.append(matrix)
    
    return matrices