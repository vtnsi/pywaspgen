import numpy as np
import pywaspgen.filters as filters

from pywaspgen.modem.digital.digital import DIGITAL

class LDAPM(DIGITAL):
    """
    Linear Digital Amplitude Phase Modulation (LDAPM) modem base class.
    """
    def __init__(self, burst):
        """
        The constructor for the base `LDAPM` class.

        Args:
            sig_type (dict): The signal type of the LDAPM modem.
            pulse_type (dict): The pulse shape metadata of the LDAPM modem.
        """
        super().__init__(burst)

        self._update_burst_metadata(burst)
        self.pulse_type = burst.metadata["pulse_type"]
        self.__set_pulse_shape()

    def _update_burst_metadata(self, burst):
        rng = np.random.default_rng(42)
        beta = round(100.0 * rng.uniform(*burst.metadata["pulse_shape"]["beta"])) / 100.0
        span = rng.integers(*burst.metadata["pulse_shape"]["span"], endpoint=True)
        sps = round(10.0 * ((beta + 1.0) / burst.bandwidth)) / 10.0
        burst.bandwidth = (beta + 1.0) / sps
        burst.metadata["pulse_type"] = {"sps": sps, "format": burst.metadata["pulse_shape"]["format"], "params": {"beta": beta, "span": span, "window": (burst.metadata["pulse_shape"]["window"]["type"], burst.metadata["pulse_shape"]["window"]["params"])}}

    def _gen_samples(self, num_samples):
        """
        Generates a random modulated LDAPM data sample stream of pulse shaped LDAPM data symbols from the LDAPM modem's data symbol table.

        Args:
            num_samples (int): The length, in samples, of the random modulated LDAPM data sample stream to generate.

        Returns:
            float complex: A numpy array, of size defined by ``num_samples``, of pulse shaped LDAPM data symbols chosen uniformly from the LDAPM modem's data symbol table.
        """
        total_symbols = self.pulse_shaper.calc_num_symbols(num_samples)
        if total_symbols >= 1:
            return self.pulse_shaper.filter(self.gen_symbols(total_symbols), "interpolate")[0:num_samples]
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