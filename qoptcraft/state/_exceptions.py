class ProbabilityError(ValueError):
    """Probabilities don't add up to 1."""

    def __init__(self, sum_probs: float) -> None:
        message = f"The sum of the probabilities is {sum_probs} instead of 1."
        super().__init__(message)


class PureStateLengthError(ValueError):
    """States and probabilities differ in length."""

    def __init__(self) -> None:
        message = "States and probabilities differ in length."
        super().__init__(message)


class NumberPhotonsError(ValueError):
    """Not all states have the same number of photons."""

    def __init__(self) -> None:
        message = "Not all states have the same number of photons."
        super().__init__(message)


class NumberModesError(ValueError):
    """Not all states have the same number of modes."""

    def __init__(self) -> None:
        message = "Not all states have the same number of modes."
        super().__init__(message)


class NotHermitianError(ValueError):
    """The matrix is not hermitian."""

    def __init__(self) -> None:
        message = "The matrix is not hermitian."
        super().__init__(message)
