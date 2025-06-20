"""
This module provides functions for adding impairments to in-phase/quadrature (IQ) data.
"""

import numpy as np

def awgn(samps, snr_db, rng=np.random.default_rng()):
    """
    Applies Additive White Gaussian Noise (AWGN) to the IQ data ``samps``. Assumes the IQ data is normalized to unit average power.

    Args:
        samps (float complex): The IQ data to apply AWGN to.
        snr_db (float): The signal-to-noise ratio (SNR), in dB, of the returned IQ data.
        rng (obj): A numpy random generator object used by the random generators.

    Returns:
        float complex: The input IQ data, ``samps``, impaired by AWGN with a SNR (in dB) specified by ``snr_db``.
    """
    
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_samps = rng.normal(0.0, np.sqrt(0.5), len(samps)) + 1.0j * (rng.normal(0.0, np.sqrt(0.5), len(samps)))
    return np.sqrt(snr_lin / 2.0) * samps + noise_samps

def freq_off(samps, freq):
    """
    Applies a complex frequency shift to the IQ data.

    Args:
        samps (float complex): The IQ data to apply a complex frequency shift to.
        freq (float): The amount of frequency by which to shift the IQ data by.

    Returns:
        float complex: The input IQ data, ``samps``, impaired by a complex frequency shift specified by ``freq``.
    """
    return samps * [np.exp(2.0j * np.pi * freq * x) for x in range(0, len(samps))]
