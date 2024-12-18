import numpy as np
from numpy.typing import ArrayLike
# import jax.numpy as np  # uncomment to use JAX


def compute_crlb(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 1
) -> np.ndarray:
    """Compute the Cramer-Rao Lower Bound (CRLB) for phase linking estimation.

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex (true) coherence matrix (N x N)

    num_looks : int
        Number of looks used in estimation


    Returns
    -------
    crlb : np.ndarray
        Covariance matrix (N-1 x N-1) representing the CRLB

    """
    N = coherence_matrix.shape[0]

    # Create basis matrix B to handle rank deficiency
    # This maps N phases to N-1 phase differences
    B = np.zeros((N, N - 1))
    B[1:, :] = np.eye(N - 1)

    # Compute absolute coherence matrix, and its inverse
    abs_coherence = np.abs(coherence_matrix)
    inv_abs_coherence = np.linalg.inv(abs_coherence)

    R_aps_inv = np.eye(N) * 1 / aps_variance

    # Compute Fisher Information Matrix
    # FIM_ij = 2L|γ_ij|²/(1-|γ_ij|²)
    X = 2 * num_looks * (abs_coherence * inv_abs_coherence - np.eye(N))

    # Project FIM to N-1 space and invert to get covariance
    # projected_fim = B.T @ (X + np.eye(N)) @ B
    projected_fim = B.T @ (X + R_aps_inv) @ B
    crlb = np.linalg.inv(projected_fim)

    # Ensure symmetry (handle numerical issues)
    return (crlb + crlb.T) / 2


def compute_lower_bound_std(coherence_matrix: ArrayLike, num_looks: int) -> np.ndarray:
    """Compute the lower bound (in degrees) on the standard deviation of the phase linking estimator.

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex (true) coherence matrix (N x N)

    num_looks : int
        Number of looks used in estimation

    Returns
    -------
    lower_bound_std : np.ndarray
        Lower bound on the standard deviation of the phase linking estimator

    """
    crlb = compute_crlb(coherence_matrix=coherence_matrix, num_looks=num_looks)

    estimator_stddev = np.sqrt(np.diag(crlb))
    return np.concatenate(([0], estimator_stddev))
