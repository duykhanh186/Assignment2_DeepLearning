import numpy as np


def relative_error(a, b, eps=1e-12):
    return np.linalg.norm(a - b) / (
        np.linalg.norm(a)
        + np.linalg.norm(b)
        + eps
    )


def numerical_gradient_array(f, x, df=None, h=1e-5):
    """
    Numerical gradient for scalar or array-valued f.

    If df is None, f(x) must return a scalar.
    Otherwise computes sum(f(x) * df).
    """
    grad = np.zeros_like(x, dtype=np.float64)

    it = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + h
        pos = f(x)

        x[idx] = old - h
        neg = f(x)

        x[idx] = old

        if df is None:
            grad[idx] = (pos - neg) / (2 * h)
        else:
            grad[idx] = np.sum((pos - neg) * df) / (2 * h)

        it.iternext()

    return grad
