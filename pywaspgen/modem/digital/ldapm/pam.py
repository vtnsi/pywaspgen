import numpy as np
from scipy.special import erfc

from pywaspgen.modem.digital.ldapm.ldapm import LDAPM

class PAM(LDAPM):
    """
    Pulse Amplitude Modulation (PAM) modem class.
    """
    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the `PAM` class.

        Args:
            sig_type (dict): The signal type of the PAM modem.
            pulse_type (dict): The pulse shape metadata of the PAM modem.
        """
        super().__init__(sig_type, pulse_type)

    def _symbol_table_create(self):
        """
        Creates the modem's PAM data symbol table.
        """
        self.symbol_table = []
        for k in range(1, int(self.order / 2) + 1):
            self.symbol_table.append(-2.0 * k + 1.0)
            self.symbol_table.append(2.0 * k - 1.0)
        self.symbol_table = self.symbol_table/np.sqrt(np.mean(np.abs(self.symbol_table)**2.0))


    def _get_theory_awgn(self, snr_lin):
        """
        Calculates the theoretical symbol error rate of the PAM modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_lin (float): The signal-to-noise ratio (SNR) to get the symbol error rate for.

        Returns:
            float: The theoretical symbol error rate in an AWGN channel of the PAM modem for an SNR, in dB, specified by ``snr_db``.
        """              
        return (2.0 * (self.order - 1.0) / self.order) * (1.0 / 2.0) * erfc(np.sqrt((6.0 * snr_lin) / (self.order**2.0 - 1.0)) / np.sqrt(2.0))