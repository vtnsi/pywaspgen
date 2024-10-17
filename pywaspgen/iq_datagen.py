"""
This module provides functionality for generating in-phase/quadrature(IQ) data from bursts via the :class:`IQDatagen` object.
"""

import json
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from pywaspgen import impairments, modems, validate_schema


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
            validate_schema.validate_schema(instance)
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
        modem = modems.LDAPM(burst.sig_type, burst.metadata["pulse_type"])
        samples = modem.gen_samples(burst.duration, burst.metadata["snr"])
        if samples.size != 0:
            samples = impairments.freq_off(samples, burst.cent_freq)
        return samples, modem

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

        iq_data = impairments.awgn(np.zeros(self.config["spectrum"]["observation_duration"], dtype=np.csingle))

        for k in range(len(burst_list)):
            snr = rng.uniform(self.config["iq_defaults"]["snr"][0], self.config["iq_defaults"]["snr"][1])
            beta = round(100.0 * rng.uniform(self.config["pulse_shape_defaults"]["beta"][0], self.config["pulse_shape_defaults"]["beta"][1])) / 100.0
            span = rng.integers(self.config["pulse_shape_defaults"]["span"][0], self.config["pulse_shape_defaults"]["span"][1], endpoint=True)
            sps = round(10.0 * ((beta + 1.0) / burst_list[k].bandwidth)) / 10.0
            burst_list[k].bandwidth = (beta + 1.0) / sps
            burst_list[k].metadata["snr"] = snr
            burst_list[k].metadata["pulse_type"] = {
                "sps": sps,
                "format": self.config["pulse_shape_defaults"]["format"],
                "params": {
                    "beta": beta,
                    "span": span,
                    "window": (
                        self.config["pulse_shape_defaults"]["window"]["type"],
                        self.config["pulse_shape_defaults"]["window"]["params"],
                    ),
                },
            }

        new_burst_list = []
        for burst in burst_list:
            samples, modem = self._get_iq(burst)
            start_iq_idx = max(0, burst.start)
            start_samples_idx = max(0, -burst.start)
            num_samples = min(burst.duration - start_samples_idx, self.config["spectrum"]["observation_duration"] - start_iq_idx)
            iq_data[start_iq_idx : start_iq_idx + num_samples] += samples[start_samples_idx : start_samples_idx + num_samples]

            burst.start = start_iq_idx
            burst.duration = num_samples

            if self.config["spectrum"]["save_modems"]:
                burst.metadata["modem"] = modem
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
