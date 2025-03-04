import numpy as np

from pywaspgen.modem.modem import MODEM


class DIGITAL(MODEM):
    """
    Digital modem base class.
    """

    def __init__(self, burst):
        """
        The constructor for the base `DIGITAL` class.

        Args:
            sig_type (dict): The signal type of the digital modem.
        """
        super().__init__(burst)
        self.order = burst.sig_type["order"]
        self.symbol_table_create()

    def gen_symbols(self, num_symbols):
        """
        Generates a random set of symbols from the digital modem's symbol table.

        Args:
            num_symbols (int): The number of random data symbols to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_symbols``, of random data symbols chosen uniformly from the digital modem's data symbol table.
        """
        self.generated_symbols_idx = np.random.randint(0, len(self.symbol_table), num_symbols)
        self.generated_symbols = []
        for symbol_idx in self.generated_symbols_idx:
            self.generated_symbols.append(self.symbol_table[symbol_idx])
        return self.generated_symbols

    def get_symbols(self, samples):
        """
        Demodulates a modulated data sample stream.

        Args:
            samples (float complex): The modulated data symbols to be demodulated.

        Returns:
            float complex: A numpy array of demodulated data symbols calculated from ``samples``.
        """
        return self.__get_symbols(samples)

    def set_sps(self, sps):
        """
        Sets the samples per symbol of the digital modem.

        Args:
            sps (float): The samples per symbol to be used by the digital modem for upsampling the data symbols.
        """
        self.pulse_type["sps"] = sps
        self.__set_pulse_shape()

    def symbol_table_create(self):
        """
        Creates the modem's IQ data symbol table of the type specified by ``DIGITAL.sig_type``.
        """
        self.symbol_table = []
        self._symbol_table_create()

    def _get_sim_awgn(self, samples):
        """
        Calculates a simulated error rate of the digital modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            samples (float complex): The modulated IQ data symbols to calculate the error rate from.

        Returns:
            float: The simulated error rate in an AWGN channel of the digital modem calculated from ``samples``.
        """
        demoded_symbols = self._get_symbols(samples)

        symbol_errors = 0
        for k in range(len(demoded_symbols)):
            symbol_errors = symbol_errors + (1 - np.equal(self.generated_symbols_idx[k], demoded_symbols[k]))
        return symbol_errors / len(demoded_symbols)
