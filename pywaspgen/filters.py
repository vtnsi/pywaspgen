"""
This module provides classes for resampling and pulseshaping in-phase/quadrature (IQ) data.
"""

import math
from fractions import Fraction

import numpy as np
import scipy.signal as signal


class _Filter:
    """
    Base class for performing pulse shape filtering of IQ data.
    """

    def __init__(
        self,
        pulse_type={
            "sps": 2,
            "format": "RRC",
            "params": {"beta": 0.35, "span": 10, "window": ("kaiser", 5.0)},
        },
    ):
        """
        The constructor for the `_Filter` class.

        Args:
            pulse_type (dict): The pulse shape metadata of the burst.
        """
        self.pulse_type = pulse_type
        self.ratio = Fraction(self.pulse_type["sps"]).limit_denominator()
        self.up = self.ratio.numerator
        self.down = self.ratio.denominator
        self._gen_taps()

    def __upsample(self, data):
        """
        Upsamples the input IQ data.

        Args:
            data (float complex): The IQ data to upsample.

        Returns:
            float complex: The downsampled ``data``.
        """
        upsamp_data = np.zeros(self.up * len(data)).astype(np.cdouble)
        upsamp_data[:: self.up] = data
        return upsamp_data

    def __downsample(self, data):
        """
        Downsamples the input IQ data.

        Args:
            data (float complex): The IQ data to downsample.

        Returns:
            float complex: The upsampled ``data``.
        """
        cutoff = len(data) - self.delay
        return data[self.delay - 1 : cutoff : self.up]

    def filter(self, data, type):
        """
        Applies either a interpolation or decimation filter to the input IQ data using a specified pulse shaping filter.

        Args:
            data (float complex): The IQ data to apply the filtering to.
            type (str): Specifies whether an 'interpolation' or 'decimation' filter is applied to the input IQ data.

        Returns:
            float complex: The input IQ data, ``data``, after filtering of type defined by the pulse shaping filter of type ``self.pulse_type``.
        """
        if type == "interpolate":
            up_samps = np.convolve(self.__upsample(data), self.taps, "full")
            down_samps = signal.resample_poly(up_samps, 1, self.down, window=self.pulse_type["params"]["window"])
            return down_samps / (np.sqrt(np.mean(np.abs(down_samps) ** 2.0) * (self.pulse_type["sps"] / 2.0)))
        elif type == "decimate":
            up_samps = signal.resample_poly(data, self.down, 1, window=self.pulse_type["params"]["window"])
            conv_samps = np.convolve(up_samps, self.taps, "full")
            down_samps = self.__downsample(conv_samps)
            if self.pulse_type["sps"] % 1 == 0:
                return down_samps / (np.sqrt(np.mean(np.abs(down_samps) ** 2.0)))
            else:
                return down_samps[0:-1] / (np.sqrt(np.mean(np.abs(down_samps) ** 2.0)))

    def calc_num_symbols(self, num_samples):
        """
        Calculates the number of modulated IQ data symbols needed to achieve a specified number of output samples after applying an interpolation filter.

        Args:
            num_samples (int): The number of IQ samples wanted after filtering.

        Returns:
            int: The number of modulated IQ data symbols needed to achieve the amount of IQ samples specified by ``num_samples`` after applying an interpolation filter.
        """
        return int(np.ceil((num_samples - self.pulse_type["params"]["span"] * self.pulse_type["sps"] + 1.0) / self.pulse_type["sps"]))


class RRC(_Filter):
    """
    Root-Raised Cosine (RRC) pulse shaping filter class.
    """

    def __init__(self, pulse_type):
        """
        The constructor for the `RRC` class.

        Args:
            pulse_type (dict): The pulse shape metadata of the burst.
        """
        super().__init__(pulse_type)

    def _gen_taps(self):
        """
        Returns:
            float: The RRC filter taps to be used for filtering.
        """
        beta = self.pulse_type["params"]["beta"]

        self.taps = []
        for k in range(
            -int(self.pulse_type["params"]["span"] * self.up / 2.0),
            int(self.pulse_type["params"]["span"] * self.up / 2.0) + 1,
        ):
            if k == 0:
                self.taps.append((1.0 / (np.sqrt(self.up))) * ((1.0 - beta) + ((4.0 * beta) / np.pi)))
            elif math.isclose(abs(k), self.up / (4.0 * beta)):
                self.taps.append((beta / (np.sqrt(2.0 * self.up))) * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))))
            else:
                self.taps.append((1.0 / (np.sqrt(self.up))) * ((np.sin((np.pi * k * (1.0 - beta)) / self.up) + (((4.0 * beta * k) / self.up) * np.cos((np.pi * k * (1.0 + beta)) / self.up))) / (((np.pi * k) / self.up) * (1.0 - ((4.0 * beta * k) / self.up) ** 2.0))))
        self.delay = len(self.taps)
