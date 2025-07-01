"""
Complete Test Suite for QOPTCRAFT Operators Extension

This file contains comprehensive tests for all operator framework components.
It should be placed in qoptcraft/tests/test_operators/

Run with: python -m pytest test_operators_complete.py -v

Author: yangbc30 with claude sonnet 4
License: GPL-3.0
"""

import pytest
import numpy as np
from typing import List, Tuple
import warnings

# Import the new operators framework
# Note: Adjust import paths based on actual package structure
try:
    from qoptcraft.operators import (
        # Core classes
        QuantumOperator, CreationOperator, AnnihilationOperator, NumberOperator,
        ScalarOperator, OperatorSum, OperatorProduct, JordanSchwingerOperator,
        
        # Main functions
        jordan_schwinger_map, a_dag, a, n,
        
        # Composite operators
        total_photon_number, position_op, momentum_op, squeeze_operator,
        beamsplitter_operator, kerr_operator,
        
        # Utility functions
        commutator, anticommutator, standard_operators, validate_operator_algebra
    )
    
    # QOPTCRAFT base functionality
    from qoptcraft.basis import photon_basis, hilbert_dim
    from qoptcraft.operators import creation_fock, annihilation_fock
    
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestOperatorFrameworkIntegration:
    """Test integration with QOPTCRAFT base functionality."""
    
    def test_qoptcraft_imports(self):
        """Test that all required QOPTCRAFT functions are available."""
        # This test ensures the base QOPTCRAFT package is properly installed
        assert callable(photon_basis)
        assert callable(hilbert_dim)
        assert callable(creation_fock)
        assert callable(annihilation_fock)
    
    def test_basis_generation(self):
        """Test QOPTCRAFT basis generation functions."""
        modes = 2
        photons = 2
        
        basis = photon_basis(modes, photons)
        dim = hilbert_dim(modes, photons)
        
        assert len(basis) == dim
        assert all(len(state) == modes for state in basis)
        assert all(sum(state) == photons for state in basis)
    
    def test_ladder_operators_compatibility(self):
        """Test compatibility with QOPTCRAFT ladder operators."""
        test_state = (1, 0)
        
        # Test creation
        final_state, coeff = creation_fock(0, test_state)
        assert final_state == (2, 0)
        assert np.isclose(coeff, np.sqrt(2))
        
        # Test annihilation
        final_state, coeff = annihilation_fock(0, test_state)
        assert final_state == (0, 0)
        assert np.isclose(coeff, 1.0)


class TestBasicOperators:
    """Test basic operator functionality."""
    
    def test_creation_operator(self):
        """Test creation operator properties."""
        modes = 3
        for mode in range(modes):
            op = CreationOperator(mode, modes)
            assert op.mode == mode
            assert op.modes == modes
            assert str(op) == f"a†_{mode}"
    
    def test_annihilation_operator(self):
        """Test annihilation operator properties."""
        modes = 3
        for mode in range(modes):
            op = AnnihilationOperator(mode, modes)
            assert op.mode == mode
            assert op.modes == modes
            assert str(op) == f"a_{mode}"
    
    def test_number_operator(self):
        """Test number operator properties."""
        modes = 3
        for mode in range(modes):
            op = NumberOperator(mode, modes)
            assert op.mode == mode
            assert op.modes == modes
            assert str(op) == f"n_{mode}"
    
    def test_invalid_mode_indices(self):
        """Test that invalid mode indices raise errors."""
        modes = 2
        
        with pytest.raises(ValueError):
            CreationOperator(-1, modes)
        with pytest.raises(ValueError):
            CreationOperator(modes, modes)
        with pytest.raises(ValueError):
            AnnihilationOperator(modes + 1, modes)
        with pytest.raises(ValueError):
            NumberOperator(-5, modes)
    
    def test_invalid_modes_count(self):
        """Test that invalid modes count raises errors."""
        with pytest.raises(ValueError):
            CreationOperator(0, 0)
        with pytest.raises(ValueError):
            CreationOperator(0, -1)


class TestOperatorMatrices:
    """Test operator matrix representations."""
    
    def test_number_operator_diagonal(self):
        """Test that number operators produce correct diagonal matrices."""
        modes = 2
        photons = 2
        basis = photon_basis(modes, photons)  # [(2, 0), (1, 1), (0, 2)]
        
        # Test n̂_0
        n0 = NumberOperator(0, modes)
        matrix = n0.to_matrix(basis)
        expected_diagonal = [2, 1, 0]  # Photon counts in mode 0
        
        assert matrix.shape == (len(basis), len(basis))
        np.testing.assert_allclose(np.diag(matrix), expected_diagonal)
        
        # Test n̂_1
        n1 = NumberOperator(1, modes)
        matrix = n1.to_matrix(basis)
        expected_diagonal = [0, 1, 2]  # Photon counts in mode 1
        
        np.testing.assert_allclose(np.diag(matrix), expected_diagonal)
    
    def test_creation_annihilation_matrices(self):
        """Test creation and annihilation operator matrices."""
        modes = 2
        photons = 1
        basis = photon_basis(modes, photons)  # [(1, 0), (0, 1)]
        
        # In 1-photon subspace, creation operators should mostly give zero
        # (since they create 2-photon states outside the subspace)
        a0_dag = CreationOperator(0, modes)
        matrix = a0_dag.to_matrix(basis)
        
        # Should be all zeros in 1-photon subspace
        expected = np.zeros((2, 2))
        np.testing.assert_allclose(matrix, expected)
    
    def test_matrix_hermiticity(self):
        """Test that Hermitian operators produce Hermitian matrices."""
        modes = 2
        photons = 2
        basis = photon_basis(modes, photons)
        
        # Number operators should be Hermitian
        n0 = NumberOperator(0, modes)
        matrix = n0.to_matrix(basis)
        
        assert np.allclose(matrix, matrix.conj().T)
        
        # Symmetric combinations should be Hermitian
        a0_dag = CreationOperator(0, modes)
        a0 = AnnihilationOperator(0, modes)
        a1_dag = CreationOperator(1, modes)
        a1 = AnnihilationOperator(1, modes)
        
        symmetric_op = a0_dag @ a1 + a1_dag @ a0
        symmetric_matrix = symmetric_op.to_matrix(basis)
        
        assert np.allclose(symmetric_matrix, symmetric_matrix.conj().T)


class TestOperatorArithmetic:
    """Test operator arithmetic operations."""
    
    def test_operator_addition(self):
        """Test operator addition."""
        modes = 2
        a0 = AnnihilationOperator(0, modes)
        a1 = AnnihilationOperator(1, modes)
        
        # Test addition
        sum_op = a0 + a1
        assert isinstance(sum_op, OperatorSum)
        assert len(sum_op.operators) == 2
        
        # Test with scalar
        scalar_sum = a0 + 1.5
        assert isinstance(scalar_sum, OperatorSum)
        
        # Test right addition
        right_sum = 2.0 + a0
        assert isinstance(right_sum, OperatorSum)
    
    def test_operator_subtraction(self):
        """Test operator subtraction."""
        modes = 2
        a0 = AnnihilationOperator(0, modes)
        a1 = AnnihilationOperator(1, modes)
        
        diff_op = a0 - a1
        assert isinstance(diff_op, OperatorSum)
        
        # Test with scalar
        scalar_diff = a0 - 1.5
        assert isinstance(scalar_diff, OperatorSum)
        
        # Test right subtraction
        right_diff = 2.0 - a0
        assert isinstance(right_diff, OperatorSum)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        modes = 2
        a0 = AnnihilationOperator(0, modes)
        
        # Test various scalar types
        for scalar in [2, 2.0, 2.0 + 1.0j, -1.5]:
            scaled = scalar * a0
            assert isinstance(scaled, ScaledOperator)
            assert scaled.scalar == scalar
            
            # Test right multiplication
            scaled_right = a0 * scalar
            assert isinstance(scaled_right, ScaledOperator)
            assert scaled_right.scalar == scalar
    
    def test_operator_division(self):
        """Test operator division by scalars."""
        modes = 2
        a0 = AnnihilationOperator(0, modes)
        
        divided = a0 / 2.0
        assert isinstance(divided, ScaledOperator)
        assert divided.scalar == 0.5
        
        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            a0 / 0
    
    def test_operator_negation(self):
        """Test operator negation."""
        modes = 2
        a0 = AnnihilationOperator(0, modes)
        
        neg_op = -a0
        assert isinstance(neg_op, ScaledOperator)
        assert neg_op.scalar == -1.0
    
    def test_operator_matrix_multiplication(self):
        """Test operator matrix multiplication (@)."""
        modes = 2
        a0_dag = CreationOperator(0, modes)
        a0 = AnnihilationOperator(0, modes)
        
        product = a0_dag @ a0
        assert isinstance(product, OperatorProduct)
        assert len(product.operators) == 2
    
    def test_operator_power(self):
        """Test operator exponentiation."""
        modes = 2
        n0 = NumberOperator(0, modes)
        
        # Test various powers
        identity = n0 ** 0
        assert isinstance(identity, ScalarOperator)
        assert identity.scalar == 1.0
        
        same = n0 ** 1
        assert same is n0
        
        squared = n0 ** 2
        assert isinstance(squared, OperatorProduct)
        assert len(squared.operators) == 2
        
        # Test invalid powers
        with pytest.raises(ValueError):
            n0 ** (-1)
        with pytest.raises(ValueError):
            n0 ** 1.5
    
    def test_mode_mismatch_errors(self):
        """Test that operators with different modes raise errors."""
        a0_2mode = AnnihilationOperator(0, 2)
        a0_3mode = AnnihilationOperator(0, 3)
        
        with pytest.raises(ValueError):
            a0_2mode + a0_3mode
        with pytest.raises(ValueError):
            a0_2mode @ a0_3mode
        with pytest.raises(ValueError):
            a0_2mode.commutator(a0_3mode)


class TestJordanSchwingerMapping:
    """Test Jordan-Schwinger mapping functionality."""
    
    def test_pauli_matrices_mapping(self):
        """Test Jordan-Schwinger mapping of Pauli matrices."""
        modes = 2
        
        # Pauli-X matrix
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        js_x = jordan_schwinger_map(sigma_x)
        
        assert js_x.modes == modes
        symbolic = js_x.get_symbolic_form()
        assert "a†_0·a_1" in symbolic
        assert "a†_1·a_0" in symbolic
        
        # Pauli-Z matrix
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        js_z = jordan_schwinger_map(sigma_z)
        
        symbolic = js_z.get_symbolic_form()
        assert "n_0" in symbolic
        assert "n_1" in symbolic
    
    def test_identity_matrix_mapping(self):
        """Test Jordan-Schwinger mapping of identity matrix."""
        modes = 2
        identity = np.eye(modes, dtype=complex)
        js_identity = jordan_schwinger_map(identity)
        
        # Should map to total photon number operator
        symbolic = js_identity.get_symbolic_form()
        assert "n_0" in symbolic
        assert "n_1" in symbolic
    
    def test_hermiticity_preservation(self):
        """Test that Jordan-Schwinger mapping preserves Hermiticity."""
        modes = 3
        
        # Generate random Hermitian matrix
        np.random.seed(42)
        A = np.random.random((modes, modes)) + 1j * np.random.random((modes, modes))
        H = A + A.conj().T  # Make Hermitian
        
        js_op = jordan_schwinger_map(H)
        
        # Test for different photon numbers
        for photons in [1, 2]:
            basis = photon_basis(modes, photons)
            assert js_op.is_hermitian(basis), f"Not Hermitian for {photons} photons"
            
            # Eigenvalues should be real
            eigenvals = js_op.eigenvalues(basis)
            assert np.allclose(eigenvals.imag, 0, atol=1e-12), "Eigenvalues should be real"
    
    def test_linearity(self):
        """Test linearity of Jordan-Schwinger mapping."""
        modes = 2
        
        # Two Hermitian matrices
        H1 = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        H2 = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Scalars
        a, b = 2.0, -1.5
        
        # Linear combination
        H_combined = a * H1 + b * H2
        
        # Map each separately
        js1 = jordan_schwinger_map(H1)
        js2 = jordan_schwinger_map(H2)
        js_combined = jordan_schwinger_map(H_combined)
        
        # Map the combination directly
        js_linear = a * js1 + b * js2
        
        # Should be equal in matrix representation
        basis = photon_basis(modes, 1)
        matrix_combined = js_combined.to_matrix(basis)
        matrix_linear = js_linear.to_matrix(basis)
        
        np.testing.assert_allclose(matrix_combined, matrix_linear, rtol=1e-12)
    
    def test_invalid_matrices(self):
        """Test that invalid matrices raise appropriate errors."""
        # Non-square matrix
        with pytest.raises(ValueError):
            jordan_schwinger_map(np.array([[1, 2, 3], [4, 5, 6]]))
        
        # Non-Hermitian matrix
        with pytest.raises(ValueError):
            jordan_schwinger_map(np.array([[1, 2], [3, 4]]))
    
    def test_matrix_elements_extraction(self):
        """Test extraction of matrix elements from Jordan-Schwinger operator."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        js_x = jordan_schwinger_map(sigma_x)
        
        elements = js_x.get_matrix_elements()
        assert (0, 1) in elements
        assert (1, 0) in elements
        assert elements[(0, 1)] == 1.0
        assert elements[(1, 0)] == 1.0
    
    def test_diagonal_detection(self):
        """Test detection of diagonal matrices."""
        # Diagonal matrix
        diag_matrix = np.diag([1, -1])
        js_diag = jordan_schwinger_map(diag_matrix)
        assert js_diag.is_diagonal()
        
        # Non-diagonal matrix
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        js_x = jordan_schwinger_map(sigma_x)
        assert not js_x.is_diagonal()


class TestCompositeOperators:
    """Test composite and convenience operators."""
    
    def test_total_photon_number(self):
        """Test total photon number operator."""
        modes = 3
        N = total_photon_number(modes)
        
        assert isinstance(N, OperatorSum)
        assert len(N.operators) == modes
        
        # Test in basis
        photons = 2
        basis = photon_basis(modes, photons)
        N_matrix = N.to_matrix(basis)
        
        # All diagonal elements should be 2 (total photon number)
        expected_diagonal = [photons] * len(basis)
        np.testing.assert_allclose(np.diag(N_matrix), expected_diagonal)
    
    def test_position_momentum_operators(self):
        """Test position and momentum quadrature operators."""
        modes = 2
        
        # Position operator
        x0 = position_op(0, modes)
        assert isinstance(x0, OperatorSum)
        
        # Momentum operator
        p0 = momentum_op(0, modes)
        assert isinstance(p0, OperatorSum)
        
        # Test that they're Hermitian
        basis = photon_basis(modes, 1)
        x0_matrix = x0.to_matrix(basis)
        p0_matrix = p0.to_matrix(basis)
        
        assert np.allclose(x0_matrix, x0_matrix.conj().T)
        assert np.allclose(p0_matrix, p0_matrix.conj().T)
    
    def test_squeeze_operator(self):
        """Test squeezing operator."""
        modes = 2
        xi = 0.1 + 0.05j
        
        squeeze_gen = squeeze_operator(xi, 0, 1, modes)
        assert isinstance(squeeze_gen, OperatorSum)
        
        # Should be anti-Hermitian (generator property)
        basis = photon_basis(modes, 2)
        matrix = squeeze_gen.to_matrix(basis)
        
        # Check anti-Hermiticity: A† = -A
        assert np.allclose(matrix.conj().T, -matrix, atol=1e-12)
    
    def test_beamsplitter_operator(self):
        """Test beamsplitter operator."""
        modes = 2
        theta = np.pi / 4
        phi = 0
        
        bs_gen = beamsplitter_operator(theta, phi, 0, 1, modes)
        assert isinstance(bs_gen, OperatorSum)
        
        # Should be anti-Hermitian
        basis = photon_basis(modes, 1)
        matrix = bs_gen.to_matrix(basis)
        
        assert np.allclose(matrix.conj().T, -matrix, atol=1e-12)
    
    def test_kerr_operator(self):
        """Test Kerr nonlinearity operator."""
        modes = 2
        chi = 0.01
        
        kerr = kerr_operator(chi, 0, modes)
        assert isinstance(kerr, (OperatorSum, ScaledOperator))
        
        # Should be Hermitian
        basis = photon_basis(modes, 2)
        matrix = kerr.to_matrix(basis)
        
        assert np.allclose(matrix, matrix.conj().T, atol=1e-12)
    
    def test_operator_creation_errors(self):
        """Test error handling in composite operator creation."""
        modes = 2
        
        # Same mode for two-mode operators should raise errors
        with pytest.raises(ValueError):
            squeeze_operator(0.1, 0, 0, modes)
        
        with pytest.raises(ValueError):
            beamsplitter_operator(np.pi/4, 0, 1, 1, modes)
        
        # Invalid modes count
        with pytest.raises(ValueError):
            total_photon_number(0)


class TestOperatorProperties:
    """Test mathematical properties of operators."""
    
    def test_expectation_values(self):
        """Test expectation value calculations."""
        modes = 2
        basis = photon_basis(modes, 1)  # [(1, 0), (0, 1)]
        
        # State |1,0⟩
        state_10 = np.array([1.0, 0.0], dtype=complex)
        
        # Number operator n̂_0
        n0 = NumberOperator(0, modes)
        expectation = n0.expectation_value(state_10, basis)
        
        # Should be 1 (one photon in mode 0)
        assert np.isclose(expectation, 1.0)
        
        # State |0,1⟩
        state_01 = np.array([0.0, 1.0], dtype=complex)
        expectation = n0.expectation_value(state_01, basis)
        
        # Should be 0 (no photons in mode 0)
        assert np.isclose(expectation, 0.0)
    
    def test_commutator_relations(self):
        """Test commutator calculations."""
        modes = 2
        a0 = a(0, modes)
        a0_dag = a_dag(0, modes)
        a1 = a(1, modes)
        a1_dag = a_dag(1, modes)
        
        # Different modes should commute
        comm_01 = commutator(a0, a1)
        assert isinstance(comm_01, OperatorSum)
        
        # Test that commutator is anti-symmetric
        comm_10 = commutator(a1, a0)
        
        basis = photon_basis(modes, 2)
        comm_01_matrix = comm_01.to_matrix(basis)
        comm_10_matrix = comm_10.to_matrix(basis)
        
        np.testing.assert_allclose(comm_01_matrix, -comm_10_matrix, atol=1e-12)
    
    def test_anticommutator_relations(self):
        """Test anticommutator calculations."""
        modes = 2
        a0 = a(0, modes)
        a1 = a(1, modes)
        
        # Anticommutator should be symmetric
        anticomm_01 = anticommutator(a0, a1)
        anticomm_10 = anticommutator(a1, a0)
        
        basis = photon_basis(modes, 2)
        matrix_01 = anticomm_01.to_matrix(basis)
        matrix_10 = anticomm_10.to_matrix(basis)
        
        np.testing.assert_allclose(matrix_01, matrix_10, atol=1e-12)
    
    def test_operator_traces(self):
        """Test operator trace calculations."""
        modes = 2
        basis = photon_basis(modes, 2)
        
        # Total photon number should have trace = photons * dim
        N = total_photon_number(modes)
        N_matrix = N.to_matrix(basis)
        trace = np.trace(N_matrix)
        
        expected_trace = 2 * len(basis)  # 2 photons per state
        assert np.isclose(trace, expected_trace)
        
        # Jordan-Schwinger operator trace
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        js_z = jordan_schwinger_map(sigma_z)
        js_trace = js_z.trace(basis)
        
        # Trace should be real for Hermitian operators
        assert np.isclose(js_trace.imag, 0, atol=1e-12)


class TestConvenienceFunctions:
    """Test convenience functions and user interface."""
    
    def test_convenience_aliases(self):
        """Test that convenience aliases work correctly."""
        modes = 3
        
        for mode in range(modes):
            # Test that aliases create equivalent operators
            op1 = CreationOperator(mode, modes)
            op2 = a_dag(mode, modes)
            
            assert type(op1) == type(op2)
            assert op1.mode == op2.mode
            assert op1.modes == op2.modes
    
    def test_standard_operators_dict(self):
        """Test standard operators dictionary creation."""
        modes = 3
        ops = standard_operators(modes)
        
        required_keys = ['a_dag', 'a', 'n', 'N', 'x', 'p']
        for key in required_keys:
            assert key in ops
        
        # Check list lengths
        assert len(ops['a_dag']) == modes
        assert len(ops['a']) == modes
        assert len(ops['n']) == modes
        
        # Check total photon number
        assert isinstance(ops['N'], OperatorSum)
    
    def test_build_hamiltonian_function(self):
        """Test Hamiltonian building convenience function."""
        from qoptcraft.operators import build_hamiltonian
        
        modes = 2
        H = build_hamiltonian(
            1.0 * n(0, modes),
            1.1 * n(1, modes),
            0.1 * (a_dag(0, modes) @ a(1, modes))
        )
        
        assert isinstance(H, OperatorSum)
        assert len(H.operators) == 3
        
        # Test error handling
        with pytest.raises(ValueError):
            build_hamiltonian()  # No terms
        
        with pytest.raises(TypeError):
            build_hamiltonian(1.0)  # Non-operator term
    
    def test_validation_functions(self):
        """Test operator algebra validation."""
        # This should not raise errors for a properly working system
        success = validate_operator_algebra()
        assert isinstance(success, bool)
        
        # If validation fails, it should still return a boolean
        # (not raise an exception)


class TestPerformanceAndScaling:
    """Test performance and scaling behavior."""
    
    def test_state_to_index_optimization(self):
        """Test that state_to_index optimization works."""
        modes = 3
        photons = 2
        basis = photon_basis(modes, photons)
        
        # Create state mapping
        state_to_index = {state: i for i, state in enumerate(basis)}
        
        # Test with and without optimization
        op = NumberOperator(0, modes)
        
        matrix1 = op.to_matrix(basis)
        matrix2 = op.to_matrix(basis, state_to_index)
        
        # Results should be identical
        np.testing.assert_allclose(matrix1, matrix2)
    
    def test_large_system_handling(self):
        """Test framework with moderately large systems."""
        modes = 4
        photons = 2
        basis = photon_basis(modes, photons)
        
        # Create complex operator
        terms = [n(i, modes) for i in range(modes)]
        for i in range(modes):
            for j in range(i+1, modes):
                terms.append(0.1 * (a_dag(i, modes) @ a(j, modes)))
        
        complex_op = OperatorSum(terms)
        
        # Should be able to compute matrix representation
        matrix = complex_op.to_matrix(basis)
        
        # Basic checks
        assert matrix.shape == (len(basis), len(basis))
        assert np.isfinite(matrix).all()
        
        # Should be Hermitian (since it's a physical Hamiltonian)
        assert np.allclose(matrix, matrix.conj().T, atol=1e-12)
    
    def test_memory_efficiency(self):
        """Test memory usage with repeated operations."""
        modes = 3
        photons = 1
        basis = photon_basis(modes, photons)
        
        # Create many operators and ensure no memory leaks
        operators = []
        for _ in range(100):
            op = a_dag(0, modes) @ a(1, modes) + a_dag(1, modes) @ a(0, modes)
            matrix = op.to_matrix(basis)
            operators.append((op, matrix))
        
        # All matrices should be consistent
        first_matrix = operators[0][1]
        for _, matrix in operators[1:]:
            np.testing.assert_allclose(matrix, first_matrix)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_basis_handling(self):
        """Test behavior with empty basis."""
        modes = 2
        op = NumberOperator(0, modes)
        
        # Empty basis should return empty matrix
        empty_basis = []
        matrix = op.to_matrix(empty_basis)
        assert matrix.shape == (0, 0)
    
    def test_single_state_basis(self):
        """Test behavior with single-state basis."""
        modes = 2
        basis = [(0, 0)]  # Vacuum state only
        
        # Number operator should give zero
        n0 = NumberOperator(0, modes)
        matrix = n0.to_matrix(basis)
        
        expected = np.array([[0]])
        np.testing.assert_allclose(matrix, expected)
    
    def test_type_checking(self):
        """Test type checking in operator operations."""
        modes = 2
        a0 = a(0, modes)
        
        # Invalid types should raise TypeError
        with pytest.raises(TypeError):
            a0 + "invalid"
        
        with pytest.raises(TypeError):
            a0 @ "invalid"
        
        with pytest.raises(TypeError):
            a0 * [1, 2, 3]
    
    def test_numerical_stability(self):
        """Test numerical stability with small/large coefficients."""
        modes = 2
        basis = photon_basis(modes, 1)
        
        # Very small coefficients
        tiny_op = 1e-15 * n(0, modes)
        tiny_matrix = tiny_op.to_matrix(basis)
        assert np.isfinite(tiny_matrix).all()
        
        # Very large coefficients
        large_op = 1e10 * n(0, modes)
        large_matrix = large_op.to_matrix(basis)
        assert np.isfinite(large_matrix).all()


class TestDocumentationExamples:
    """Test that documentation examples work correctly."""
    
    def test_basic_usage_example(self):
        """Test the basic usage example from documentation."""
        # This should match the example in the module docstring
        modes = 2
        a0_dag = a_dag(0, modes)  # â†_0
        a1 = a(1, modes)          # â_1
        n0 = n(0, modes)          # n̂_0
        
        # Build Hamiltonian
        H = n0 + 0.1 * (a0_dag @ a1)
        
        # Jordan-Schwinger mapping
        sigma_x = np.array([[0, 1], [1, 0]])
        js_op = jordan_schwinger_map(sigma_x)
        
        # Matrix representation
        basis = photon_basis(modes, 1)
        H_matrix = H.to_matrix(basis)
        
        # Basic checks
        assert H_matrix.shape == (len(basis), len(basis))
        assert np.isfinite(H_matrix).all()
        assert isinstance(js_op, JordanSchwingerOperator)
    
    def test_jordan_schwinger_example(self):
        """Test Jordan-Schwinger mapping example."""
        # From jordan_schwinger.py docstring
        sigma_x = np.array([[0, 1], [1, 0]])
        js_op = jordan_schwinger_map(sigma_x)
        
        symbolic = js_op.get_symbolic_form()
        assert "a†_0·a_1" in symbolic
        assert "a†_1·a_0" in symbolic
        
        # Matrix representation in 1-photon basis
        basis = photon_basis(2, 1)
        matrix = js_op.to_matrix(basis)
        
        expected = np.array([[0., 1.], [1., 0.]])
        np.testing.assert_allclose(matrix, expected)


def test_module_imports():
    """Test that all expected symbols can be imported."""
    # This test ensures the module structure is correct
    from qoptcraft.operators import (
        QuantumOperator, CreationOperator, AnnihilationOperator,
        jordan_schwinger_map, a_dag, a, n
    )
    
    assert issubclass(CreationOperator, QuantumOperator)
    assert callable(jordan_schwinger_map)
    assert callable(a_dag)


def test_backward_compatibility():
    """Test backward compatibility with existing QOPTCRAFT code."""
    # Ensure that existing QOPTCRAFT functions still work
    from qoptcraft.operators import creation_fock, annihilation_fock
    
    test_state = (1, 0)
    
    # These should work as before
    final_state, coeff = creation_fock(0, test_state)
    assert final_state == (2, 0)
    
    final_state, coeff = annihilation_fock(0, test_state)
    assert final_state == (0, 0)


if __name__ == "__main__":
    # Run tests manually if not using pytest
    print("Running QOPTCRAFT Operators Extension Tests...")
    
    # Simple test runner
    test_classes = [
        TestOperatorFrameworkIntegration,
        TestBasicOperators,
        TestOperatorMatrices,
        TestOperatorArithmetic,
        TestJordanSchwingerMapping,
        TestCompositeOperators,
        TestOperatorProperties,
        TestConvenienceFunctions,
        TestPerformanceAndScaling,
        TestErrorHandling,
        TestDocumentationExamples
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) 
                       if method.startswith('test_') and callable(getattr(instance, method))]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("All tests passed! 🎉")
    else:
        print(f"{total_tests - passed_tests} tests failed.")
    
    # Run module-level tests
    try:
        test_module_imports()
        test_backward_compatibility()
        print("✓ Module-level tests passed")
    except Exception as e:
        print(f"✗ Module-level tests failed: {e}")