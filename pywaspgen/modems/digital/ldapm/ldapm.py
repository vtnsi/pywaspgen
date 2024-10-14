import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import pywaspgen.modems.digital.digital as digital

class LDAPM(digital.DIGITAL):
    """
    Linear Digital Amplitude Phase Modulation (LDAPM) modem class.
    """

    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the `LDAPM` class.

        Args:
            sig_type (dict): The signal type of the LDAPM modem.
            pulse_type (dict): The pulse shape metadata of the LDAPM modem.
        """
        super().__init__(sig_type, pulse_type)
        self.order = sig_type["order"]

    def _symbol_table_create(self):
        """
        Creates the modem's IQ data symbol table of the type specified by ``LDAPM.sig_type``.
        """
        self.symbol_table = []
        self.__symbol_table_create(self)

    def gen_symbols(self, num_symbols):
        """
        Generates a random set of symbols from the modem's IQ data symbol table.

        Args:
            num_symbols (int): The number of random IQ data symbols to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_symbols``, of random IQ data symbols chosen uniformly from the modem's IQ data symbol table.
        """
        self.generated_symbols = np.random.choice(self.symbol_table, num_symbols)
        return self.generated_symbols

    def get_samples(self, symbols):
        """
        Modulates an IQ data symbols stream.

        Args:
            symbols (float complex): The IQ data symbols to be modulated.

        Returns:
            float complex: A numpy array of modulated IQ data symbols provided by ``symbols``.
        """
        return self.pulse_shaper.filter(symbols, "interpolate")

    def gen_samples(self, num_samples):
        """
        Generates a random modulated IQ data sample stream of pulse shaped IQ data symbols from the modem's IQ data symbol table.

        Args:
            num_samples (int): The length, in samples, of the random modulated IQ data sample stream to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_samples``, of pulse shaped IQ data symbols chosen uniformly from the modem's IQ data symbol table.
        """
        total_symbols = self.pulse_shaper.calc_num_symbols(num_samples)
        if total_symbols >= 1:
            samples = self.pulse_shaper.filter(self.gen_symbols(total_symbols), "interpolate")
            return samples[0:num_samples]
        else:
            return np.array([])

    def get_symbols(self, samples):
        """
        Demodulates a modulated IQ data sample stream.

        Args:
            samples (float complex): The modulated IQ data symbols to be demodulated.

        Returns:
            float complex: A numpy array of demodulated IQ data symbols calculated from ``samples``.
        """
        return np.array(self.pulse_shaper.filter(samples, "decimate"))

    def get_nearest_symbol(self, symbol):
        """
        Determines the nearest symbol of the modem's IQ data symbol table to the provided input symbol.

        Args:
            symbol (float complex): The IQ symbol to find the nearest symbol to.

        Returns:
            float complex: The symbol of the modem's IQ data symbol table nearest to ``symbol``.
        """
        idx = np.abs(symbol - self.symbol_table).argmin()
        return self.symbol_table[idx]