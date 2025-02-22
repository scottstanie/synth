import numpy as np
import pytest

# TODO: use synth's simulate_coh
from dolphin.phase_link.simulate import simulate_coh

from synth.covariance import compute_crlb_batch
from synth.crlb import compute_lower_bound_std


def generate_coh_array():
    """Generate a batch of coherent matrices for testing.

    This function creates 12 simulated coherence matrices with size 5x5 using the
    simulate_coh function, and reshapes them into a batch array of shape (4, 3, 5, 5).
    """
    num_matrices = 12
    # Generate list of coherence matrices with gamma_inf=0 and varying Tau0
    Cs = [
        simulate_coh(5, gamma_inf=0, Tau0=10 * ii)[0]
        for ii in range(1, num_matrices + 1)
    ]
    Cs_arr = np.array(Cs)
    return Cs_arr.reshape(3, 4, 5, 5)


@pytest.mark.parametrize("num_looks", [1, 2])
def test_compute_crlb_batch_matches_single(num_looks):
    """Test that the batch version of CRLB calculation matches the single matrix version.

    For each pixel in the batch, the test computes the lower bound standard deviation using
    the single matrix function and compares it with the corresponding element from the batch
    computation.
    """
    C_arrays = generate_coh_array()

    # Compute batch CRLB standard deviations from the batch function
    batch_result = compute_crlb_batch(C_arrays, num_looks)
    assert batch_result.shape == (5, 3, 4)
    n, rows, cols = batch_result.shape

    for i in range(rows):
        for j in range(cols):
            single_crlb = compute_lower_bound_std(C_arrays[i, j], num_looks)
            # Use a tolerance for floating point comparisons
            errmsg = f"Mismatch at pixel ({i},{j}): single {single_crlb} vs batch {batch_result[:, i, j]}"
            assert np.allclose(single_crlb, batch_result[:, i, j], atol=1e-6), errmsg
