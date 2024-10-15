import numpy as np
from scipy.special import erfc

from pywaspgen.modem.digital.ldapm.ldapm import LDAPM

class QAM(LDAPM):
    """
    Quadrature Amplitude Modulation (QAM) modem class.
    """
    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the `QAM` class.

        Args:
            sig_type (dict): The signal type of the QAM modem.
            pulse_type (dict): The pulse shape metadata of the QAM modem.
        """
        super().__init__(sig_type, pulse_type)

    def _symbol_table_create(self):
        """
        Creates the modem's QAM data symbol table.
        """    
        self.symbol_table = []    
        max_val = int(np.sqrt(self.order) - 1.0)
        for k in range(-max_val, max_val + 1, 2):
            for kk in range(-max_val, max_val + 1, 2):
                self.symbol_table.append(k + 1.0j * kk)

    def _get_theory_awgn(self, snr_lin):
        """
        Calculates the theoretical symbol error rate of the QAM modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_lin (float): The signal-to-noise ratio (SNR) to get the symbol error rate for.

        Returns:
            float: The theoretical symbol error rate in an AWGN channel of the QAM modem for an SNR, in dB, specified by ``snr_db``.
        """              
        val = np.sqrt(1.0 / ((2.0 / 3.0) * (self.order - 1.0)))
        return 2.0 * (1.0 - 1.0 / np.sqrt(self.order)) * erfc(val * np.sqrt(snr_lin)) - (1.0 - 2.0 / np.sqrt(self.order) + 1.0 / self.order) * erfc(val * np.sqrt(snr_lin)) ** 2.0