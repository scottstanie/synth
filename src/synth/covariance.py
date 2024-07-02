"""Module for simulating stacks of SLCs to test phase linking algorithms.

Contains simple versions of MLE and EVD estimator to compare against the
full CPU/GPU stack implementations.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from numpy.typing import ArrayLike, NDArray

rng = np.random.default_rng()


def ccg_noise(N: int) -> NDArray[np.complex64]:
    """Create N samples of standard complex circular Gaussian noise."""
    return (
        rng.normal(scale=1 / np.sqrt(2), size=2 * N)
        .astype(np.float32)
        .view(np.complex64)
    )


@partial(jit, static_argnums=(1,))
def ccg_noise_jax(key: random.PRNGKey, N: int) -> Array:
    """Create N samples of standard complex circular Gaussian noise using JAX.

    Parameters
    ----------
    key : random.PRNGKey
        JAX random key for generating random numbers.
    N : int
        Number of complex samples to generate.

    Returns
    -------
    jnp.ndarray
        Array of N complex64 samples.

    Notes
    -----
    This function generates complex circular Gaussian noise with zero mean
    and unit variance. The real and imaginary parts are independent and
    each has a standard deviation of 1/sqrt(2).

    """
    # Split the key for generating real and imaginary parts
    key_real, key_imag = random.split(key)

    # Generate real and imaginary parts separately
    real_part = random.normal(key_real, shape=(N,), dtype=jnp.float32) / jnp.sqrt(2.0)
    imag_part = random.normal(key_imag, shape=(N,), dtype=jnp.float32) / jnp.sqrt(2.0)

    # Combine real and imaginary parts into complex numbers
    return real_part + 1j * imag_part


def simulate_coh_stack(
    time: np.ndarray,
    gamma0: np.ndarray,
    gamma_inf: np.ndarray,
    Tau0: np.ndarray,
    signal: np.ndarray | None = None,
    seasonal_mask: np.ndarray | None = None,
    seasonal_A: np.ndarray | None = None,
    seasonal_B: np.ndarray | None = None,
) -> np.ndarray:
    """Create a coherence matrix at each pixel.

    Parameters
    ----------
    time : np.ndarray
        Array of time values, arbitrary units.
        shape (N,) where N is the number of acquisitions.
    gamma0 : np.ndarray
        Initial coherence value for each pixel.
    gamma_inf : np.ndarray
        Asymptotic coherence value for each pixel.
    Tau0 : np.ndarray
        Coherence decay time constant for each pixel.
    signal : np.ndarray
        Simulated signal phase for each pixel.
    seasonal_mask : np.ndarray, optional
        Boolean mask to indicate, for each 2D pixel, whether to
        use a seasonally decorelating model:
        gamma = (A + B cos(2pi * t / 365))
    seasonal_A : np.ndarray, optional
        A coefficient for pixels whose `seasonal_mask=True`
    seasonal_B : np.ndarray, optional
        B coefficient for pixels whose `seasonal_mask=True`

    Returns
    -------
    np.ndarray
        The simulated coherence matrix for each pixel.

    """
    num_time = time.shape[0]
    temp_baselines = jnp.abs(time[None, :] - time[:, None])
    temp_baselines = temp_baselines[None, None, :, :]
    gamma0 = jnp.atleast_2d(gamma0)[:, :, None, None]
    gamma_inf = jnp.atleast_2d(gamma_inf)[:, :, None, None]
    Tau0 = jnp.atleast_2d(Tau0)[:, :, None, None]

    gamma = (gamma0 - gamma_inf) * jnp.exp(-temp_baselines / Tau0) + gamma_inf
    if signal is not None:
        phase_diff = signal[:, None] - signal[None, :]
        phase_term = jnp.exp(1j * phase_diff)
    else:
        phase_term = jnp.exp(1j * 0)

    if seasonal_mask is not None:
        if seasonal_A is None or seasonal_B is None:
            raise ValueError(
                "Must provide `seasonal_A` and `seasonal_B` if passing `seasonal_mask`"
            )
        if seasonal_mask.dtype != bool:
            raise ValueError("`seasonal_mask` must be a boolean array")
        A = jnp.atleast_2d(seasonal_A)[:, :, None, None]
        B = jnp.atleast_2d(seasonal_B)[:, :, None, None]
        mask = jnp.atleast_2d(seasonal_mask)[:, :]
        # A, B = seasonal_coeffs.A, seasonal_coeffs.B
        # assert A.ndim == B.ndim == 4
        seasonal_factor = (A + B * jnp.cos(2 * jnp.pi * temp_baselines / 365.25)) ** 2
        # Ensure it is a valid coherence multiplier
        seasonal_factor = jnp.clip(seasonal_factor, 0, 1)
        # Where
        seasonal_factor = seasonal_factor.at[~mask, :, :].set(1.0)
        gamma = gamma * seasonal_factor

    C = gamma * phase_term

    rl, cl = jnp.tril_indices(num_time, k=-1)

    C = C.at[..., rl, cl].set(
        jnp.conj(jnp.transpose(C, axes=(0, 1, 3, 2))[..., rl, cl])
    )

    # Reset the diagonals of each pixel to 1
    rs, cs = jnp.diag_indices(num_time)
    C = C.at[:, :, rs, cs].set(1)

    return C


def _get_diffs(stack: ArrayLike) -> np.ndarray:
    """Create all differences between the deformation stack.

    Parameters
    ----------
    stack : np.ndarray
        Signal stack of shape (num_time, rows, cols).

    Returns
    -------
    np.ndarray, complex64
        Covariance phases of shape (rows, cols, num_time, num_time).

    """
    # Create all possible differences using broadcasting
    # shape: (num_time, 1, rows, cols)
    stack_i = np.exp(1j * stack[:, None, :, :])
    # shape: (1, num_time, rows, cols)
    stack_j = np.exp(1j * stack[None, :, :, :])
    diff_stack = stack_i * stack_j.conj()

    # Reshape to (rows, cols, num_time, num_time)
    return diff_stack.transpose(2, 3, 0, 1)


def make_noisy_samples(
    C: ArrayLike,
    defo_stack: ArrayLike,
    amplitudes: ArrayLike | None = None,
) -> np.ndarray:
    """Create noisy deformation samples given a covariance matrix and deformation stack.

    Parameters
    ----------
    C : ArrayLike,
        Covariance matrix of shape (num_time, num_time) , or you can pass
        on matrix per pixel as shape (rows, cols, num_time, num_time).
    defo_stack : ArrayLike,
        Deformation stack of shape (num_time, rows, cols).
    amplitudes: 2D ArrayLike, optional
        If provided, set the amplitudes of the output pixels.
        Default is to use all ones.

    Returns
    -------
    samps3d: np.ndarray
        Noisy deformation samples of shape (num_time, rows, cols).

    """
    if amplitudes is not None and amplitudes.ndim != 2:
        raise ValueError("`amplitudes` must be 2D, or None")

    num_time, rows, cols = defo_stack.shape
    shape2d = (rows, cols)
    num_pixels = np.prod(shape2d)

    C_tiled = np.tile(C, (*shape2d, 1, 1)) if C.squeeze().ndim == 2 else C

    if C_tiled.shape != (rows, cols, num_time, num_time):
        raise ValueError(f"{C_tiled.shape=}, but {defo_stack.shape=}")
    # Reset the diagonals of each pixel to 1
    rs, cs = np.diag_indices(num_time)
    C_tiled[:, :, rs, cs] = 1

    signal_cov = _get_diffs(defo_stack)
    C_tiled_with_signal = C_tiled * signal_cov
    C_unstacked = C_tiled_with_signal.reshape(num_pixels, num_time, num_time)

    noise = ccg_noise(num_time * num_pixels)
    noise_unstacked = noise.reshape(num_pixels, num_time, 1)

    L_unstacked = np.linalg.cholesky(C_unstacked)
    samps = L_unstacked @ noise_unstacked

    samps3d = np.moveaxis(samps.reshape(*shape2d, num_time), -1, 0)
    if amplitudes is None:
        return samps3d

    return samps3d * amplitudes[None, :, :]


@partial(jit, static_argnums=(3, 4))
def _sample(C_tiled, defo_stack, key, num_pixels, num_time):
    # signal_cov = _get_diffs(defo_stack)
    stack_i = jnp.exp(1j * defo_stack[:, None, :, :])
    # shape: (1, num_time, rows, cols)
    stack_j = jnp.exp(1j * defo_stack[None, :, :, :])
    diff_stack = stack_i * stack_j.conj()
    signal_cov = diff_stack.transpose(2, 3, 0, 1)
    C_tiled_with_signal = C_tiled * signal_cov
    C_unstacked = C_tiled_with_signal.reshape(num_pixels, num_time, num_time)

    # noise = ccg_noise(num_time * num_pixels)
    noise = ccg_noise_jax(key, num_time * num_pixels)
    noise_unstacked = noise.reshape(num_pixels, num_time, 1)

    L_unstacked = jnp.linalg.cholesky(C_unstacked)
    return L_unstacked @ noise_unstacked


def make_noisy_samples_jax(
    key,
    C: ArrayLike,
    defo_stack: ArrayLike,
    amplitudes: ArrayLike | None = None,
) -> np.ndarray:
    """Create noisy deformation samples given a covariance matrix and deformation stack.

    Parameters
    ----------
    key : jax.random.KeyArray
        JAX pseudo random key.
        https://jax.readthedocs.io/en/latest/_autosummary/jax.random.key.htm
    C : ArrayLike,
        Covariance matrix of shape (num_time, num_time) , or you can pass
        on matrix per pixel as shape (rows, cols, num_time, num_time).
    defo_stack : ArrayLike,
        Deformation stack of shape (num_time, rows, cols).
    amplitudes: 2D ArrayLike, optional
        If provided, set the amplitudes of the output pixels.
        Default is to use all ones.

    Returns
    -------
    samps3d: np.ndarray
        Noisy deformation samples of shape (num_time, rows, cols).

    """
    if amplitudes is not None and amplitudes.ndim != 2:
        raise ValueError("`amplitudes` must be 2D, or None")

    num_time, rows, cols = defo_stack.shape
    shape2d = (rows, cols)
    num_pixels = jnp.prod(jnp.array(shape2d))

    C_tiled = jnp.tile(C, (*shape2d, 1, 1)) if C.squeeze().ndim == 2 else jnp.asarray(C)

    if C_tiled.shape != (rows, cols, num_time, num_time):
        raise ValueError(f"{C_tiled.shape=}, but {defo_stack.shape=}")
    # Reset the diagonals of each pixel to 1
    rs, cs = jnp.diag_indices(num_time)
    C_tiled = C_tiled.at[:, :, rs, cs].set(1.0 + 0.0j)

    samps = _sample(C_tiled, defo_stack, key, int(num_pixels), int(num_time))

    samps3d = jnp.moveaxis(samps.reshape(*shape2d, num_time), -1, 0)
    if amplitudes is None:
        return samps3d

    return samps3d * amplitudes[None, :, :]
