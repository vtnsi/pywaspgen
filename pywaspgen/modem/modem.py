class MODEM:
    """
    Base modem class.
    """
    def __init__(self, sig_type):
        """
        The constructor for the base `MODEM` class.

        Args:
            sig_type (dict): The signal type of the modem.
        """
        self.sig_type = sig_type

    def gen_samples(self, num_samples=5):
        return self._gen_samples(num_samples)

    def get_sim_awgn(self, samples):             
        return self._get_sim_awgn(samples)
    
    def get_theory_awgn(self, snr_db):
        """
        Calculates the theoretical error rate of the modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_db (float): The signal-to-noise ratio (SNR), in dB, to get the error rate for.

        Returns:
            float: The theoretical error rate in an AWGN channel of the modem for an SNR, in dB, specified by ``snr_db``.
        """              
        snr_lin = 10.0**(snr_db/10.0)
        return self._get_theory_awgn(snr_lin)