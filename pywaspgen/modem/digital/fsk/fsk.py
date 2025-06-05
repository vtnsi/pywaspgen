import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from pywaspgen.modem.digital.digital import DIGITAL


class FSK(DIGITAL):
    """
    Frequency Shift Keying (FSK) modem base class.
    """

    def __init__(self, deviation, sps, sig_type):
        """
        The constructor for the base `FSK` class.

        Args:
            deviation (float): The minimum difference between the carrier frequency and the instantaneous frequency of the FSK symbols.
            sps (float): The samples per symbol to be used by the digital modem for upsampling the data symbols.
            sig_type (dict): The signal type of the LDAPM modem.
        """
        self.deviation = deviation
        self.sps = sps

        self.Rs = 1 / sps
        super().__init__(sig_type)

    def _gen_samples(self, num_samples):
        """
        Generates a random modulated FSK data sample stream from the FSK modem's data symbol table.

        Args:
            num_samples (int): The length, in samples, of the random modulated FSK data sample stream to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_samples``, of FSK data symbols chosen uniformly from the FSK modem's data symbol table.
        """
        total_symbols = int(num_samples * self.Rs) + 1
        if total_symbols >= 1:
            symbols = self.gen_symbols(total_symbols)
            t = list(range(len(symbols) * self.sps))
            samples = []
            for k in range(len(symbols)):
                samples = np.concatenate((samples, [np.exp(2.0j * np.pi * symbols[k] * x) for x in t[k * self.sps : (k + 1) * self.sps]]))
            return (samples / np.sqrt(self.sps / (2.0 * np.log2(self.order))))[0:num_samples]
        else:
            return np.array([])

    def integrand(self, y, snr_lin):
        return ((1.0 - (1.0 - (0.5 * erfc(y / np.sqrt(2)))) ** (self.order - 1)) * np.exp(-0.5 * (y - np.sqrt(2 * np.log2(self.order) * snr_lin)) ** 2.0)) / np.sqrt(2 * np.pi)

    def _get_symbols(self, samples):
        """
        Demodulates a modulated FSK data sample stream.

        Args:
            samples (float complex): The modulated FSK data symbols to be demodulated.

        Returns:
            float complex: A numpy array of demodulated FSK data symbols calculated from ``samples``.
        """
        t = list(range(len(samples)))
        demoded_symbols = np.zeros(len(self.generated_symbols))
        for k in range(len(self.generated_symbols)):
            metric = np.zeros(self.order, dtype=np.csingle)
            for kk in range(self.order):
                metric[kk] = np.vdot([np.exp(2.0j * np.pi * (self.symbol_table[kk]) * x) for x in t[k * self.sps : (k + 1) * self.sps]], samples[k * self.sps : (k + 1) * self.sps])
            demoded_symbols[k] = np.argmax(metric)
        return demoded_symbols

    def _get_theory_awgn(self, snr_lin):
        """
        Calculates the theoretical symbol error rate of the FSK modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_lin (float): The signal-to-noise ratio (SNR) to get the symbol error rate for.

        Returns:
            float: The theoretical symbol error rate in an AWGN channel of the FSK modem for an SNR, in dB, specified by ``snr_db``.
        """
        return quad(self.integrand, -np.inf, np.inf, args=(snr_lin))[0]

    def _symbol_table_create(self):
        """
        Creates the modem's FSK data symbol table.
        """
        for k in range(1, int(self.sig_type["order"] / 2) + 1):
            self.symbol_table.append((-2.0 * k + 1.0) * self.deviation)
            self.symbol_table.append((2.0 * k - 1.0) * self.deviation)
