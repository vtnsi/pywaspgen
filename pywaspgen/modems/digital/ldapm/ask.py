import numpy as np
from scipy.stats import norm

import pywaspgen.modems.digital.ldapm.ldapm as ldapm

class ASK(ldapm.LDAPM):
    """
    Amplitude Shift Keying (ASK) modem class.
    """
    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the `ASK` class.

        Args:
            sig_type (dict): The signal type of the ASK modem.
            pulse_type (dict): The pulse shape metadata of the ASK modem.
        """
        super().__init__(sig_type, pulse_type)

    def __symbol_table_create(self):
        """
        Creates the modem's ASK data symbol table.
        """        
        for k in range(0, self.order):
            self.symbol_table.append(k)

    def __get_theory_awgn(self, snr_lin):
        """
        Calculates the theoretical symbol error rate of the ASK modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_lin (float): The signal-to-noise ratio (SNR) to get the symbol error rate for.

        Returns:
            float: The theoretical symbol error rate in an AWGN channel of the ASK modem for an SNR, in dB, specified by ``snr_db``.
        """        
        if self.order == 2:
            return norm.sf(np.sqrt(2.0 * snr_lin))
        elif self.order == 4:
            return 1.0 - (1.0 - norm.sf(np.sqrt(snr_lin))) ** 2.0
        else:
            return 2.0 * norm.sf(np.sqrt(2.0 * snr_lin) * np.sin(np.pi / self.order))