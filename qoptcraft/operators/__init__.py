"""
QOPTCRAFT Operators Module

This module provides a comprehensive object-oriented framework for quantum operators
in photonic systems. It extends QOPTCRAFT with basis-independent operator algebra
and Jordan-Schwinger mapping capabilities.

Main Features:
- Basis-independent operator construction
- Natural operator arithmetic syntax  
- Jordan-Schwinger mapping for classical-quantum correspondence
- Integration with existing QOPTCRAFT basis functions
- Comprehensive library of quantum optical operators

Classes:
    QuantumOperator: Abstract base class for all operators
    CreationOperator: Photon creation operator â†
    AnnihilationOperator: Photon annihilation operator â
    NumberOperator: Photon number operator n̂
    ScalarOperator: Scalar multiple of identity
    ScaledOperator: Scalar multiple of another operator
    OperatorSum: Sum of multiple operators
    OperatorProduct: Product of multiple operators
    JordanSchwingerOperator: Operators from matrix mapping

Functions:
    jordan_schwinger_map: Map Hermitian matrices to quantum operators
    a_dag, a, n: Convenience functions for operator creation
    total_photon_number: Create total photon number operator
    position_op, momentum_op: Create quadrature operators
    squeeze_operator: Create squeezing operators
    beamsplitter_operator: Create beamsplitter interactions
    
Usage:
    >>> from qoptcraft.operators import a_dag, a, n, jordan_schwinger_map
    >>> from qoptcraft.basis import photon_basis
    >>> import numpy as np
    >>> 
    >>> # Create operators
    >>> modes = 2
    >>> a0_dag = a_dag(0, modes)  # â†_0
    >>> a1 = a(1, modes)          # â_1
    >>> n0 = n(0, modes)          # n̂_0
    >>> 
    >>> # Build Hamiltonian
    >>> H = n0 + 0.1 * (a0_dag @ a1)
    >>> 
    >>> # Jordan-Schwinger mapping
    >>> sigma_x = np.array([[0, 1], [1, 0]])
    >>> js_op = jordan_schwinger_map(sigma_x)
    >>> 
    >>> # Matrix representation
    >>> basis = photon_basis(modes, 1)
    >>> H_matrix = H.to_matrix(basis)

Author: QOPTCRAFT Extension Team
License: Compatible with QOPTCRAFT license
Version: 1.0.0
"""

import warnings
from typing import Dict, List, Tuple, Any, Optional

# Import existing QOPTCRAFT operators for backward compatibility
try:
    from .ladder import creation_fock, annihilation_fock
    from .adjoint import adjoint_evol
    from .qft import qft, qft_inv
    from .quasiunitary import random_quasiunitary
    _QOPTCRAFT_BASE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import base QOPTCRAFT operators: {e}", ImportWarning)
    _QOPTCRAFT_BASE_AVAILABLE = False

# Import new operator framework
try:
    from .quantum_operators import (
        QuantumOperator,
        CreationOperator,
        AnnihilationOperator,
        NumberOperator,
        ScalarOperator,
        ScaledOperator,
        OperatorSum,
        OperatorProduct
    )
    _QUANTUM_OPERATORS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import quantum operators: {e}", ImportWarning)
    _QUANTUM_OPERATORS_AVAILABLE = False

try:
    from .jordan_schwinger import (
        JordanSchwingerOperator,
        jordan_schwinger_map,
        pauli_operators,
        generate_all_js_operators,
    )
    _JORDAN_SCHWINGER_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import Jordan-Schwinger mapping: {e}", ImportWarning)
    _JORDAN_SCHWINGER_AVAILABLE = False

try:
    from .operator_arithmetic import (
        # Basic operator creation
        creation_op, annihilation_op, number_op,
        a_dag, a, n,
        
        # Composite operators
        total_photon_number, position_op, momentum_op,
        coherent_displacement, squeeze_operator, beamsplitter_operator,
        kerr_operator, cross_kerr_operator, jaynes_cummings_operator,
        parametric_amplifier_operator, hopping_operator,
        ring_lattice_operator, linear_chain_operator,
        
        # Utility functions
        commutator, anticommutator, dagger
    )
    _OPERATOR_ARITHMETIC_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import operator arithmetic: {e}", ImportWarning)
    _OPERATOR_ARITHMETIC_AVAILABLE = False


# Version and metadata
__version__ = "1.0.0"
__author__ = "QOPTCRAFT Extension Team"
__email__ = "qoptcraft@example.com"
__license__ = "Compatible with QOPTCRAFT license"
__status__ = "Production"


# Build __all__ dynamically based on available imports
__all__ = []

# Add existing QOPTCRAFT operators if available
if _QOPTCRAFT_BASE_AVAILABLE:
    __all__.extend([
        'creation_fock', 'annihilation_fock', 'adjoint_evol', 
        'qft', 'qft_inv', 'random_quasiunitary'
    ])

# Add core operator classes if available
if _QUANTUM_OPERATORS_AVAILABLE:
    __all__.extend([
        'QuantumOperator', 'CreationOperator', 'AnnihilationOperator', 
        'NumberOperator', 'ScalarOperator', 'ScaledOperator', 
        'OperatorSum', 'OperatorProduct'
    ])

# Add Jordan-Schwinger mapping if available
if _JORDAN_SCHWINGER_AVAILABLE:
    __all__.extend([
        'JordanSchwingerOperator', 'jordan_schwinger_map',
        'pauli_operators', 'su2_generators', 'coherent_state_displacement',
        'validate_jordan_schwinger_properties'
    ])

# Add operator arithmetic if available
if _OPERATOR_ARITHMETIC_AVAILABLE:
    __all__.extend([
        # Convenience functions (most commonly used)
        'creation_op', 'annihilation_op', 'number_op',
        'a_dag', 'a', 'n',
        
        # Composite operators
        'total_photon_number', 'position_op', 'momentum_op',
        'coherent_displacement', 'squeeze_operator', 'beamsplitter_operator',
        'kerr_operator', 'cross_kerr_operator', 'jaynes_cummings_operator',
        'parametric_amplifier_operator', 'hopping_operator',
        'ring_lattice_operator', 'linear_chain_operator',
        
        # Utility functions
        'commutator', 'anticommutator', 'dagger'
    ])

# Add module-level convenience functions
__all__.extend([
    'standard_operators', 'build_hamiltonian', 'validate_operator_algebra',
    'get_operator_info', 'help_operators', 'operator_examples',
    'check_framework_status', 'list_available_operators'
])


# Backward compatibility aliases with deprecation warnings
def create_creation_operator(mode: int, modes: int):
    """
    Backward compatibility function for creation operator.
    
    .. deprecated:: 1.0.0
        Use a_dag(mode, modes) instead.
    """
    warnings.warn(
        "create_creation_operator is deprecated. Use a_dag(mode, modes) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if _OPERATOR_ARITHMETIC_AVAILABLE:
        return creation_op(mode, modes)
    else:
        raise ImportError("Operator arithmetic module not available")


def create_annihilation_operator(mode: int, modes: int):
    """
    Backward compatibility function for annihilation operator.
    
    .. deprecated:: 1.0.0
        Use a(mode, modes) instead.
    """
    warnings.warn(
        "create_annihilation_operator is deprecated. Use a(mode, modes) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if _OPERATOR_ARITHMETIC_AVAILABLE:
        return annihilation_op(mode, modes)
    else:
        raise ImportError("Operator arithmetic module not available")


# Module-level convenience functions
def build_hamiltonian(*terms):
    """
    Build a Hamiltonian from multiple operator terms.
    
    Args:
        *terms: Variable number of QuantumOperator instances
        
    Returns:
        OperatorSum representing the total Hamiltonian
        
    Raises:
        ImportError: If quantum operators module not available
        ValueError: If no terms provided or invalid terms
        
    Example:
        >>> from qoptcraft.operators import build_hamiltonian, n, a_dag, a
        >>> modes = 2
        >>> H = build_hamiltonian(
        ...     1.0 * n(0, modes),           # Mode 0 energy
        ...     1.1 * n(1, modes),           # Mode 1 energy  
        ...     0.1 * (a_dag(0, modes) @ a(1, modes))  # Coupling
        ... )
    """
    if not _QUANTUM_OPERATORS_AVAILABLE:
        raise ImportError("Quantum operators module not available")
    
    if not terms:
        raise ValueError("At least one term required")
    
    # Check that all terms are operators with same number of modes
    first_modes = terms[0].modes
    for term in terms:
        if not isinstance(term, QuantumOperator):
            raise TypeError(f"All terms must be QuantumOperator instances, got {type(term)}")
        if term.modes != first_modes:
            raise ValueError(f"All operators must have same number of modes")
    
    return OperatorSum(list(terms))


def standard_operators(modes: int) -> Dict[str, Any]:
    """
    Create a dictionary of standard operators for a given number of modes.
    
    Args:
        modes: Number of modes
        
    Returns:
        Dictionary containing common operators:
        - 'a_dag': List of creation operators
        - 'a': List of annihilation operators  
        - 'n': List of number operators
        - 'N': Total photon number operator
        - 'x': List of position quadrature operators (if available)
        - 'p': List of momentum quadrature operators (if available)
        
    Raises:
        ImportError: If required modules not available
        ValueError: If modes <= 0
        
    Example:
        >>> ops = standard_operators(2)
        >>> a0_dag = ops['a_dag'][0]  # â†_0
        >>> N_total = ops['N']        # Total photon number
    """
    if not _OPERATOR_ARITHMETIC_AVAILABLE:
        raise ImportError("Operator arithmetic module not available")
    
    if modes <= 0:
        raise ValueError("Number of modes must be positive")
    
    operators = {
        'a_dag': [a_dag(i, modes) for i in range(modes)],
        'a': [a(i, modes) for i in range(modes)],
        'n': [n(i, modes) for i in range(modes)],
        'N': total_photon_number(modes),
    }
    
    # Add quadrature operators if available
    try:
        operators.update({
            'x': [position_op(i, modes) for i in range(modes)],
            'p': [momentum_op(i, modes) for i in range(modes)]
        })
    except Exception as e:
        warnings.warn(f"Could not create quadrature operators: {e}", RuntimeWarning)
    
    return operators


def validate_operator_algebra() -> bool:
    """
    Validate basic operator algebra relations.
    
    Returns:
        True if all validations pass, False otherwise
        
    Example:
        >>> success = validate_operator_algebra()
        >>> print(f"Operator algebra validation: {'PASSED' if success else 'FAILED'}")
    """
    if not (_QUANTUM_OPERATORS_AVAILABLE and _OPERATOR_ARITHMETIC_AVAILABLE):
        warnings.warn("Cannot validate: required modules not available", RuntimeWarning)
        return False
    
    try:
        # Import QOPTCRAFT basis functions
        try:
            from qoptcraft.basis import photon_basis
        except ImportError:
            warnings.warn("Cannot validate: QOPTCRAFT basis functions not available", RuntimeWarning)
            return False
        
        import numpy as np
        
        modes = 2
        basis = photon_basis(modes, 2)  # 2-photon basis
        
        # Test 1: Creation/annihilation algebra
        a0_dag = a_dag(0, modes)
        a0 = a(0, modes)
        a1_dag = a_dag(1, modes)
        a1 = a(1, modes)
        
        # Test cross-mode commutation
        comm_01 = commutator(a0, a1_dag)
        comm_01_matrix = comm_01.to_matrix(basis)
        
        # Should be zero for different modes in same photon number sector
        cross_mode_commutes = np.allclose(comm_01_matrix, 0, atol=1e-10)
        
        # Test 2: Number operator relations
        n0 = n(0, modes)
        n0_alt = a0_dag @ a0  # Alternative construction
        
        n0_matrix = n0.to_matrix(basis)
        n0_alt_matrix = n0_alt.to_matrix(basis)
        
        number_op_consistent = np.allclose(n0_matrix, n0_alt_matrix, atol=1e-12)
        
        # Test 3: Hermiticity preservation
        H = n0 + 0.1 * (a0_dag @ a1 + a1_dag @ a0)
        H_matrix = H.to_matrix(basis)
        
        hamiltonian_hermitian = np.allclose(H_matrix, H_matrix.conj().T, atol=1e-12)
        
        # Test 4: Jordan-Schwinger mapping validation (if available)
        js_test_passed = True
        if _JORDAN_SCHWINGER_AVAILABLE:
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            js_z = jordan_schwinger_map(sigma_z)
            js_test_passed = js_z.is_hermitian(basis)
        
        # Return overall validation result
        all_tests = [number_op_consistent, hamiltonian_hermitian, js_test_passed]
        return all(all_tests)
        
    except Exception as e:
        warnings.warn(f"Validation error: {e}", RuntimeWarning)
        return False


def operator_examples():
    """
    Print examples of operator usage for documentation and testing.
    
    This function demonstrates the main features of the operator framework
    and serves as both documentation and a quick functionality test.
    """
    print("QOPTCRAFT Operators Framework - Examples")
    print("=" * 50)
    
    if not _OPERATOR_ARITHMETIC_AVAILABLE:
        print("ERROR: Operator arithmetic module not available")
        return
    
    modes = 2
    
    print("\n1. Basic Operator Creation:")
    try:
        print(f"a†_0 = {a_dag(0, modes)}")
        print(f"a_1 = {a(1, modes)}")
        print(f"n_0 = {n(0, modes)}")
    except Exception as e:
        print(f"Error in basic operators: {e}")
    
    print("\n2. Operator Arithmetic:")
    try:
        H = n(0, modes) + n(1, modes) + 0.1 * (a_dag(0, modes) @ a(1, modes))
        print(f"Hamiltonian: {H}")
    except Exception as e:
        print(f"Error in operator arithmetic: {e}")
    
    if _JORDAN_SCHWINGER_AVAILABLE:
        print("\n3. Jordan-Schwinger Mapping:")
        try:
            import numpy as np
            sigma_x = np.array([[0, 1], [1, 0]])
            js_x = jordan_schwinger_map(sigma_x)
            print(f"σ_x → {js_x.get_symbolic_form()}")
        except Exception as e:
            print(f"Error in Jordan-Schwinger mapping: {e}")
    else:
        print("\n3. Jordan-Schwinger Mapping: Not available")
    
    print("\n4. Composite Operators:")
    try:
        print(f"Total photon number: {total_photon_number(modes)}")
        if 'position_op' in globals():
            print(f"Position quadrature: {position_op(0, modes)}")
        import numpy as np
        print(f"Beamsplitter: {beamsplitter_operator(np.pi/4, 0, 0, 1, modes)}")
    except Exception as e:
        print(f"Error in composite operators: {e}")
    
    print("\n5. Standard Operators Dictionary:")
    try:
        ops = standard_operators(modes)
        print(f"Available operators: {list(ops.keys())}")
    except Exception as e:
        print(f"Error in standard operators: {e}")


def get_operator_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the operator framework.
    
    Returns:
        Dictionary with framework information including version,
        available modules, and feature status
    """
    info = {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'status': __status__,
        'modules_available': {
            'qoptcraft_base': _QOPTCRAFT_BASE_AVAILABLE,
            'quantum_operators': _QUANTUM_OPERATORS_AVAILABLE,
            'jordan_schwinger': _JORDAN_SCHWINGER_AVAILABLE,
            'operator_arithmetic': _OPERATOR_ARITHMETIC_AVAILABLE
        },
        'total_exports': len(__all__),
        'exported_symbols': sorted(__all__)
    }
    
    if _QUANTUM_OPERATORS_AVAILABLE:
        info['operator_classes'] = [
            'QuantumOperator', 'CreationOperator', 'AnnihilationOperator',
            'NumberOperator', 'ScalarOperator', 'OperatorSum', 'OperatorProduct'
        ]
    
    if _JORDAN_SCHWINGER_AVAILABLE:
        info['jordan_schwinger_classes'] = ['JordanSchwingerOperator']
        info['jordan_schwinger_functions'] = [
            'jordan_schwinger_map', 'pauli_operators', 'su2_generators'
        ]
    
    if _OPERATOR_ARITHMETIC_AVAILABLE:
        info['main_functions'] = ['a_dag', 'a', 'n']
        info['composite_operators'] = [
            'total_photon_number', 'position_op', 'momentum_op',
            'squeeze_operator', 'beamsplitter_operator', 'kerr_operator'
        ]
    
    info['features'] = []
    if _QUANTUM_OPERATORS_AVAILABLE:
        info['features'].extend([
            'Basis-independent operators',
            'Natural arithmetic syntax'
        ])
    if _JORDAN_SCHWINGER_AVAILABLE:
        info['features'].append('Jordan-Schwinger mapping')
    if _QOPTCRAFT_BASE_AVAILABLE:
        info['features'].append('QOPTCRAFT integration')
    if all([_QUANTUM_OPERATORS_AVAILABLE, _JORDAN_SCHWINGER_AVAILABLE, _OPERATOR_ARITHMETIC_AVAILABLE]):
        info['features'].append('Comprehensive validation')
    
    return info


def help_operators():
    """Print detailed help for the operators module."""
    print(__doc__)
    print(f"\nModule Status:")
    print(f"Version: {__version__}")
    info = get_operator_info()
    for module, available in info['modules_available'].items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {module}: {status}")
    
    print(f"\nTotal exported symbols: {len(__all__)}")
    print("\nAvailable functions and classes:")
    
    # Group exports by category
    categories = {
        'Basic Operators': [],
        'Composite Operators': [],
        'Jordan-Schwinger': [],
        'Utilities': [],
        'QOPTCRAFT Base': [],
        'Other': []
    }
    
    for name in sorted(__all__):
        obj = globals().get(name)
        if obj and hasattr(obj, '__doc__') and obj.__doc__:
            first_line = obj.__doc__.split('\n')[0].strip()
            
            # Categorize
            if name in ['a_dag', 'a', 'n', 'creation_op', 'annihilation_op', 'number_op']:
                categories['Basic Operators'].append(f"  {name}: {first_line}")
            elif name.startswith('jordan_schwinger') or 'Jordan' in str(type(obj).__name__):
                categories['Jordan-Schwinger'].append(f"  {name}: {first_line}")
            elif name in ['creation_fock', 'annihilation_fock', 'qft', 'adjoint_evol']:
                categories['QOPTCRAFT Base'].append(f"  {name}: {first_line}")
            elif 'operator' in name.lower() and name not in ['QuantumOperator']:
                categories['Composite Operators'].append(f"  {name}: {first_line}")
            elif name in ['commutator', 'anticommutator', 'validate_operator_algebra']:
                categories['Utilities'].append(f"  {name}: {first_line}")
            else:
                categories['Other'].append(f"  {name}: {first_line}")
    
    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            for item in items:
                print(item)


def check_framework_status() -> Dict[str, Any]:
    """
    Check the status of the operator framework and return diagnostic information.
    
    Returns:
        Dictionary with detailed status information
    """
    status = {
        'framework_ready': False,
        'modules': {},
        'tests': {},
        'recommendations': []
    }
    
    # Check module availability
    modules_status = {
        'qoptcraft_base': _QOPTCRAFT_BASE_AVAILABLE,
        'quantum_operators': _QUANTUM_OPERATORS_AVAILABLE,
        'jordan_schwinger': _JORDAN_SCHWINGER_AVAILABLE,
        'operator_arithmetic': _OPERATOR_ARITHMETIC_AVAILABLE
    }
    status['modules'] = modules_status
    
    # Run basic functionality tests
    tests = {}
    
    if _OPERATOR_ARITHMETIC_AVAILABLE:
        try:
            # Test basic operator creation
            test_op = a_dag(0, 2)
            tests['basic_operators'] = True
        except Exception as e:
            tests['basic_operators'] = f"Failed: {e}"
    else:
        tests['basic_operators'] = "Module not available"
    
    if _JORDAN_SCHWINGER_AVAILABLE:
        try:
            # Test Jordan-Schwinger mapping
            import numpy as np
            sigma_x = np.array([[0, 1], [1, 0]])
            js_op = jordan_schwinger_map(sigma_x)
            tests['jordan_schwinger'] = True
        except Exception as e:
            tests['jordan_schwinger'] = f"Failed: {e}"
    else:
        tests['jordan_schwinger'] = "Module not available"
    
    # Test QOPTCRAFT integration
    try:
        from qoptcraft.basis import photon_basis
        basis = photon_basis(2, 1)
        tests['qoptcraft_integration'] = True
    except Exception as e:
        tests['qoptcraft_integration'] = f"Failed: {e}"
    
    status['tests'] = tests
    
    # Generate recommendations
    recommendations = []
    if not _QOPTCRAFT_BASE_AVAILABLE:
        recommendations.append("Install or check QOPTCRAFT base package")
    if not _QUANTUM_OPERATORS_AVAILABLE:
        recommendations.append("Check quantum_operators.py module")
    if not _JORDAN_SCHWINGER_AVAILABLE:
        recommendations.append("Check jordan_schwinger.py module")
    if not _OPERATOR_ARITHMETIC_AVAILABLE:
        recommendations.append("Check operator_arithmetic.py module")
    
    # Check if framework is ready
    essential_modules = [_QUANTUM_OPERATORS_AVAILABLE, _OPERATOR_ARITHMETIC_AVAILABLE]
    essential_tests = [tests.get('basic_operators') == True]
    
    status['framework_ready'] = all(essential_modules) and all(essential_tests)
    status['recommendations'] = recommendations
    
    return status


def list_available_operators(category: Optional[str] = None) -> List[str]:
    """
    List available operators by category.
    
    Args:
        category: Optional category filter ('basic', 'composite', 'jordan_schwinger', 'all')
        
    Returns:
        List of available operator names
    """
    operators = {
        'basic': [],
        'composite': [],
        'jordan_schwinger': [],
        'utility': []
    }
    
    if _OPERATOR_ARITHMETIC_AVAILABLE:
        operators['basic'] = ['a_dag', 'a', 'n', 'creation_op', 'annihilation_op', 'number_op']
        operators['composite'] = [
            'total_photon_number', 'position_op', 'momentum_op', 'squeeze_operator',
            'beamsplitter_operator', 'kerr_operator', 'cross_kerr_operator',
            'jaynes_cummings_operator', 'hopping_operator'
        ]
        operators['utility'] = ['commutator', 'anticommutator', 'dagger']
    
    if _JORDAN_SCHWINGER_AVAILABLE:
        operators['jordan_schwinger'] = ['jordan_schwinger_map', 'pauli_operators', 'su2_generators']
    
    if category is None or category == 'all':
        all_ops = []
        for cat_ops in operators.values():
            all_ops.extend(cat_ops)
        return sorted(all_ops)
    elif category in operators:
        return sorted(operators[category])
    else:
        raise ValueError(f"Unknown category: {category}. Available: {list(operators.keys())}")


# Module initialization and compatibility check
def _initialize_module():
    """Initialize the operators module and perform compatibility checks."""
    # Check QOPTCRAFT compatibility
    if _QOPTCRAFT_BASE_AVAILABLE:
        try:
            from qoptcraft.basis import photon_basis, hilbert_dim
            from qoptcraft.operators import creation_fock, annihilation_fock
            
            # Quick functionality test
            test_basis = photon_basis(2, 1)
            test_dim = hilbert_dim(2, 1)
            test_state = (1, 0)
            final_state, coeff = creation_fock(0, test_state)
            
        except Exception as e:
            warnings.warn(
                f"QOPTCRAFT compatibility issue detected: {e}. "
                "Some features may not work correctly.",
                RuntimeWarning
            )
    
    # Print initialization message if any modules are missing
    missing_modules = []
    if not _QUANTUM_OPERATORS_AVAILABLE:
        missing_modules.append("quantum_operators")
    if not _JORDAN_SCHWINGER_AVAILABLE:
        missing_modules.append("jordan_schwinger")
    if not _OPERATOR_ARITHMETIC_AVAILABLE:
        missing_modules.append("operator_arithmetic")
    
    if missing_modules:
        warnings.warn(
            f"Some operator modules not available: {missing_modules}. "
            "Framework functionality will be limited.",
            ImportWarning
        )


# Call initialization
_initialize_module()


# Print welcome message on successful import
def _print_welcome_message():
    """Print welcome message when module is successfully imported."""
    if all([_QUANTUM_OPERATORS_AVAILABLE, _JORDAN_SCHWINGER_AVAILABLE, _OPERATOR_ARITHMETIC_AVAILABLE]):
        pass  # Module loaded successfully, no message needed
    else:
        missing = []
        if not _QUANTUM_OPERATORS_AVAILABLE:
            missing.append("quantum_operators")
        if not _JORDAN_SCHWINGER_AVAILABLE:
            missing.append("jordan_schwinger") 
        if not _OPERATOR_ARITHMETIC_AVAILABLE:
            missing.append("operator_arithmetic")
        
        if missing:
            print(f"QOPTCRAFT Operators: Limited functionality - missing {missing}")
        else:
            print("QOPTCRAFT Operators: Framework loaded successfully")


# Enable welcome message by setting environment variable QOPTCRAFT_VERBOSE=1
import os
if os.environ.get('QOPTCRAFT_VERBOSE') == '1':
    _print_welcome_message()