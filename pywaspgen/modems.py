"""
This module defines classes for generating modulators and demodulators (modems) for generating in-phase/quadrature (IQ) data of specific communication formats.
"""

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import pywaspgen.filters as filters


class LDAPM:
    """
    Linear Digital Amplitude Phase Modulation (LDAPM) modem class.
    """

    def __init__(
        self,
        sig_type={"format": "psk", "order": 4},
        pulse_type={
            "sps": 2,
            "format": "RRC",
            "params": {"beta": 0.35, "span": 10, "window": ("kaiser", 5.0)},
        },
    ):
        """
        The constructor for the `LDAPM` class.

        Args:
            sig_type (dict): The signal type of the modem.
            pulse_type (dict): The pulse shape metadata of the modem.
        """
        self.sig_type = sig_type
        self.pulse_type = pulse_type
        self.__symbol_table_create()
        self.__symbol_table_norm()
        self.__set_pulse_shape()

    def __symbol_table_create(self):
        """
        Creates the modem's IQ data symbol table of the type specified by ``LDAPM.sig_type``.
        """
        M = self.sig_type["order"]
        self.symbol_table = []
        if self.sig_type["format"] == "ask":
            for k in range(0, M):
                self.symbol_table.append(k)
        elif self.sig_type["format"] == "pam":
            for k in range(1, int(M / 2) + 1):
                self.symbol_table.append(-2.0 * k + 1.0)
                self.symbol_table.append(2.0 * k - 1.0)
        elif self.sig_type["format"] == "psk":
            self.symbol_table = [(np.cos((2.0 * np.pi * k) / M + (np.log2(M) - 1.0) * (np.pi / 4.0)) + 1.0j * np.sin((2.0 * np.pi * k) / M + (np.log2(M) - 1.0) * (np.pi / 4.0))) for k in range(0, M)]
        elif self.sig_type["format"] == "qam":
            max_val = int(np.sqrt(M) - 1.0)
            for k in range(-max_val, max_val + 1, 2):
                for kk in range(-max_val, max_val + 1, 2):
                    self.symbol_table.append(k + 1.0j * kk)

    def __symbol_table_norm(self):
        """
        Normalizes the modem's IQ data symbol table to have unit average energy.
        """
        self.symbol_table = np.divide(self.symbol_table, np.sqrt(np.mean(np.abs(self.symbol_table) ** 2.0)))

    def __set_pulse_shape(self):
        """
        Sets the pulse shaping filter of the modem based on ``LDAPM.pulse_type``.
        """
        class_method = getattr(filters, self.pulse_type["format"])
        self.pulse_shaper = class_method(self.pulse_type)

    def set_sps(self, sps):
        """
        Sets the samples per symbol of the modem.

        Args:
            sps (float): The samples per symbol to be used by the modem when performing filtering operations.
        """
        self.pulse_type["sps"] = sps
        self.__set_pulse_shape(self)

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

    def get_theory_awgn(self, snr_db):
        """
        Calculates the theoretical symbol error rate of the modem when impacted by an Additive White Gaussian Noise (AWGN) channel.

        Args:
            snr_db (float): The signal-to-noise ratio (SNR), in dB, to get the symbol error rate for.

        Returns:
            float: The theoretical symbol error rate in an AWGN channel of the modem for an SNR, in dB, specified by ``snr_db``.
        """
        M = self.sig_type["order"]
        snr_lin = 10.0 ** (snr_db / 10.0)
        if self.sig_type["format"] == "ask":
            return ((M - 1.0) / M) * erfc(np.sqrt((3.0 / ((M**2.0) - 1.0)) * (snr_lin / (2.0 * np.sqrt(np.log2(M))))))
        elif self.sig_type["format"] == "pam":
            return (2.0 * (M - 1.0) / M) * (1.0 / 2.0) * erfc(np.sqrt((6.0 * snr_lin) / (M**2.0 - 1.0)) / np.sqrt(2.0))
        elif self.sig_type["format"] == "psk":
            if M == 2:
                return norm.sf(np.sqrt(2.0 * snr_lin))
            elif M == 4:
                return 1.0 - (1.0 - norm.sf(np.sqrt(snr_lin))) ** 2.0
            else:
                return 2.0 * norm.sf(np.sqrt(2.0 * snr_lin) * np.sin(np.pi / M))
        elif self.sig_type["format"] == "qam":
            val = np.sqrt(1.0 / ((2.0 / 3.0) * (M - 1.0)))
            return 2.0 * (1.0 - 1.0 / np.sqrt(M)) * erfc(val * np.sqrt(snr_lin)) - (1.0 - 2.0 / np.sqrt(M) + 1.0 / M) * erfc(val * np.sqrt(snr_lin)) ** 2.0
