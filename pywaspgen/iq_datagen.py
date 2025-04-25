"""
This module provides functionality for generating in-phase/quadrature(IQ) data from bursts via the :class:`IQDatagen` object.
"""

import json
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from pywaspgen import modem
from pywaspgen import impairments

class IQDatagen:
    """
    Used to create in-phase/quadrature (IQ) data given burst definition objects, of type :class:`pywaspgen.burst_def.BurstDef`.
    """

    def __init__(self, config_file="configs/default.json"):
        """
        The constructor for the `IQDatagen` class.

        Args:
            config_file (str): The relative file path for the configuration file to be used for generating the IQ data.
        """
        with open(config_file, "r") as config_file_id:
            instance = json.load(config_file_id)
            self.config = instance
        self.rng = np.random.default_rng(self.config["generation"]["rand_seed"])

    def _get_iq(self, burst):
        """
        Generates IQ data for a burst definition, of type :class:`pywaspgen.burst_def.BurstDef`.

        Args:
            burst (obj): The :class:`pywaspgen.burst_def.BurstDef` object to create IQ data for.

        Returns:
            float complex, obj: The IQ data of the provided burst definition, the :class:`pywaspgen.modems` object used to create the IQ data.
        """
        if burst.sig_type["type"] == "fsk":
            Rs = burst.bandwidth / (burst.metadata["modulation_index"] * ((burst.sig_type["order"] - 1) ** 2.0) + 2.0)
            deviation = (burst.metadata["modulation_index"] * Rs * (burst.sig_type["order"] - 1)) / 2.0

            sps = int(1 / Rs)
            deviation = (burst.metadata["modulation_index"] * (1 / sps) * (burst.sig_type["order"] - 1)) / 2.0

            sig_modem = getattr(modem, burst.sig_type["type"])(deviation, sps, burst.sig_type)
        else:
            sig_modem = getattr(modem, burst.sig_type["type"])(burst.sig_type, burst.metadata['pulse_type'])

        samples = sig_modem.gen_samples(burst.duration)
        if samples.size != 0:
            samples = impairments.freq_off(samples, burst.cent_freq)
            samples = np.sqrt((10.0 ** (burst.metadata["snr"] / 10.0)) / 2.0) * samples
        return samples, sig_modem

    def gen_batch(self, burst_lists):
        """
        Generates a batch of IQ data using multiprocessing across cpu cores.

        Args:
            burst_lists (obj): A list of burst lists of :class:`pywaspgen.burst_def.BurstDef` objects.

        Returns:
            float complex: A list of numpy IQ data arrays for the provided ``burst_lists``.
        """
        with multiprocessing.Pool(self.config["generation"]["pool"]) as p:
            return list(zip(*list(p.starmap(self.gen_iqdata, tqdm.tqdm(zip(burst_lists, range(len(burst_lists))), total=len(burst_lists))))))

    def gen_iqdata(self, burst_list, data_idx=-1):
        """
        Generates a random aggregate IQ data from the provided burst_list with modem parameters randomized based on ranges specified by the configuration file.

        Args:
            burst_list (obj): The list of burst_def objects, of type :class:`pywaspgen.burst_def.BurstDef`, to create the IQ data.
            data_idx (int): Index for multiprocessing randomization bookkeeping by the gen_batch function. Not used otherwise.

        Returns:
            float complex, obj: A numpy array of aggregate IQ data generated from the ``burst_list``, An updated ``burst_list`` with values adjusted based on the parameters used to create the aggregate IQ data.
        """
        rng = self.rng if data_idx == -1 else np.random.default_rng([data_idx, self.config["generation"]["rand_seed"]])

        iq_data = impairments.awgn(np.zeros(self.config["spectrum"]["observation_duration"], dtype=np.csingle), -np.inf)

        for k in range(len(burst_list)):
            snr = rng.uniform(self.config["sig_defaults"]["iq"]["snr"][0], self.config["sig_defaults"]["iq"]["snr"][1])
            if burst_list[k].sig_type["type"] == "ask" or burst_list[k].sig_type["type"] == "psk" or burst_list[k].sig_type["type"] == "pam" or burst_list[k].sig_type["type"] == "qam":
                beta = round(100.0 * rng.uniform(self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["beta"][0], self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["beta"][1])) / 100.0
                span = rng.integers(self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["span"][0], self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["span"][1], endpoint=True)
                sps = round(10.0 * ((beta + 1.0) / burst_list[k].bandwidth)) / 10.0
                burst_list[k].bandwidth = (beta + 1.0) / sps
                burst_list[k].metadata["pulse_type"] = {"sps": sps, "format": self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["format"], "params": {"beta": beta, "span": span, "window": (self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["window"]["type"], self.config["sig_defaults"]["iq"]["ldapm"]["pulse_shape"]["window"]["params"])}}
            else:
                modulation_index = rng.uniform(self.config["sig_defaults"]["iq"]["fsk"]["modulation_index"][0], self.config["sig_defaults"]["iq"]["fsk"]["modulation_index"][1])
                burst_list[k].metadata["modulation_index"] = modulation_index
            burst_list[k].metadata["snr"] = snr

        new_burst_list = []
        for burst in burst_list:
            samples, sig_modem = self._get_iq(burst)
            print('SNR est: ', 10*np.log10(np.mean(np.abs(np.sqrt(sps)*samples)**2)))

            burst.duration = len(samples)
            start_iq_idx = max(0, burst.start)
            start_samples_idx = max(0, -burst.start)
            num_samples = min(burst.duration - start_samples_idx, self.config["spectrum"]["observation_duration"] - start_iq_idx)

            iq_data[start_iq_idx : start_iq_idx + num_samples] += samples[start_samples_idx : start_samples_idx + num_samples]

            burst.start = start_iq_idx
            burst.duration = num_samples

            if self.config["sig_defaults"]["save_modems"]:
                burst.metadata["modem"] = sig_modem
            new_burst_list.append(burst)

        return iq_data, new_burst_list

    def plot_iqdata(self, iq_data, ax=[]):
        """
        Plots the spectrogram of provided IQ data.

        Args:
            iq_data (float complex): The IQ data to compute the spectrogram of.
            ax (obj): Matplotlib axis in which to put the plot.

        Returns:
            obj: Matplotlib plot of the spectrogram of the IQ data.
        """
        if ax == []:
            fig, ax = plt.subplots()
            fig.set_size_inches(1000.0 / plt.rcParams["figure.dpi"], 500.0 / plt.rcParams["figure.dpi"])
            plt.xlim([0, self.config["spectrum"]["observation_duration"]])
            plt.ylim([-1.0 / 2.0, 1.0 / 2.0])
            plt.xlabel("Time [samples]")
            plt.ylabel("Normalized Frequency [f/F_s]")
            plt.xticks(np.linspace(0, self.config["spectrum"]["observation_duration"], 11))
            plt.yticks(np.linspace(-1.0 / 2.0, 1.0 / 2.0, 21))
            plt.title("Spectrogram River Plot")
            ax.set_facecolor("xkcd:salmon")
        else:
            ax.set_title("Spectrogram River Plot with Overlaid Spectrum Metadata")

        plt.specgram(iq_data, 256, 1.0, noverlap=64, cmap="plasma")
        plt.xlim([0, self.config["spectrum"]["observation_duration"]])
        cbar = plt.colorbar()
        cbar.set_label("Amplitude (dB)")
        return ax
