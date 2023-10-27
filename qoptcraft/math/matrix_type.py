"""Matrix type (numpy array or scipy sparse matrix)"""

from numpy.typing import NDArray
from scipy.sparse import spmatrix

Matrix = NDArray | spmatrix
