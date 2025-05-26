import pickle
from functools import wraps

from qoptcraft import config


def saved_basis(file_name: str):
    """Decorator to save the basis calculated in a function"""

    def decorator_saved_basis(basis_function):
        @wraps(basis_function)
        def wrapper(*args, **kwargs):
            cache = kwargs.get("cache", True)  # Default to True if not specified
            if not cache:
                return basis_function(*args, **kwargs)

            try:
                modes = kwargs.get("modes")
                if modes is None:
                    modes = args[0]
                photons = kwargs.get("photons")
                if photons is None:
                    photons = args[1]
            except IndexError as error:
                raise ValueError(
                    "Function must be called with 'modes' and 'photons' as first two arguments."
                ) from error

            orthonormal = kwargs.get("orthonormal", False)
            order = kwargs.get("order")
            invariant_operator = kwargs.get("invariant_operator")

            folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
            folder_path.mkdir(parents=True, exist_ok=True)

            # new variable to avoid errors because Python reuses closures
            file_name_ = file_name + "_orthonormal" if orthonormal else file_name
            if invariant_operator is not None:
                file_name_ += f"_{invariant_operator}"
            if order is not None:
                file_name_ += f"_order_{order}"
            basis_path = folder_path / (file_name_ + ".pkl")
            basis_path.touch()  # create file if it doesn't exist
            try:
                with basis_path.open("rb") as f:
                    basis = pickle.load(f)
            except EOFError:
                kwargs.update({"cache": False})
                basis = basis_function(*args, **kwargs)
                with basis_path.open("wb") as f:
                    pickle.dump(basis, f)
                print(f"Basis saved in {basis_path}")

            return basis

        return wrapper

    return decorator_saved_basis
