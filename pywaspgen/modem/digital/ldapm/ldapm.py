import numpy as np
import pywaspgen.filters as filters

from pywaspgen.modem.digital.digital import DIGITAL

class LDAPM(DIGITAL):
    """
    Linear Digital Amplitude Phase Modulation (LDAPM) modem base class.
    """
    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the base `LDAPM` class.

        Args:
            sig_type (dict): The signal type of the LDAPM modem.
            pulse_type (dict): The pulse shape metadata of the LDAPM modem.
        """
        super().__init__(sig_type)
        self.pulse_type = pulse_type
        self.__set_pulse_shape()     
        
    def _gen_samples(self, num_samples, rng):
        """
        Generates a random modulated LDAPM data sample stream of pulse shaped LDAPM data symbols from the LDAPM modem's data symbol table.

        Args:
            num_samples (int): The length, in samples, of the random modulated LDAPM data sample stream to generate.
            rng (obj): A numpy random generator object used by the random generators.

        Returns:
            float complex: A numpy array, of size defined by ``num_samples``, of pulse shaped LDAPM data symbols chosen uniformly from the LDAPM modem's data symbol table.
        """
        total_symbols = self.pulse_shaper.calc_num_symbols(num_samples)
        if total_symbols >= 1:
            return self.pulse_shaper.filter(self.gen_symbols(total_symbols, rng), "interpolate")[0:num_samples]
        else:
            return np.array([])     
    
    def _get_symbols(self, samples):
        """
        Demodulates a modulated LDAPM data sample stream.

        Args:
            samples (float complex): The modulated LDAPM data symbols to be demodulated.

        Returns:
            float complex: A numpy array of demodulated LDAPM data symbols calculated from ``samples``.
        """
        noisy_symbols = np.array(self.pulse_shaper.filter(samples, "decimate"))
        demoded_symbols = np.empty(len(noisy_symbols))
        for k in range(len(noisy_symbols)):
            demoded_symbols[k] = (np.abs(noisy_symbols[k] - self.symbol_table)).argmin()
        return demoded_symbols
    
    def __set_pulse_shape(self):
        """
        Sets the pulse shaping filter of the modem based on ``DIGITAL.pulse_type``.
        """
        class_method = getattr(filters, self.pulse_type["format"])
        self.pulse_shaper = class_method(self.pulse_type)    