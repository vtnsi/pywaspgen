import numpy as np

from pywaspgen.modem.modem import MODEM

class DIGITAL(MODEM):
    """
    Digital modem base class.
    """
    def __init__(self, sig_type):
        """
        The constructor for the base `DIGITAL` class.

        Args:
            sig_type (dict): The signal type of the digital modem.
        """
        super().__init__(sig_type)    
        self.order = sig_type["order"]    
        self.symbol_table_create()

    def gen_symbols(self, num_symbols):
        """
        Generates a random set of symbols from the digital modem's symbol table.

        Args:
            num_symbols (int): The number of random data symbols to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_symbols``, of random data symbols chosen uniformly from the digital modem's data symbol table.
        """
        self.generated_symbols = np.random.choice(self.symbol_table, num_symbols)
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

    def _get_symbols(self, samples):
        return self.__get_symbols(samples)

    def symbol_table_create(self):
        """
        Creates the modem's IQ data symbol table of the type specified by ``DIGITAL.sig_type``.
        """
        self.symbol_table = []
        self._symbol_table_create()

    def _get_sim_awgn(self, samples):             
        demoded_symbols = self._get_symbols(samples)
        
        symbol_errors = 0
        for k in range(len(demoded_symbols)):
            symbol_errors = symbol_errors + (1 - np.equal(self.generated_symbols[k], demoded_symbols[k]))
        return symbol_errors/len(demoded_symbols)