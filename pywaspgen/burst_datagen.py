"""
This module provides functionality for generating burst data via the :class:`BurstDatagen` object.
"""
import distinctipy
import json
import multiprocessing
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tqdm
import uuid
from pywaspgen import burst_def

class BurstDatagen:
    """
    Used to create burst definition objects, of type :class:`pywaspgen.burst_def.BurstDef`, that specify the time and frequency extent of a burst as well as the burst's metadata. These objects are used as input to :class:`pywaspgen.iq_datagen.IQDatagen` to create in-phase/quadrature (IQ) data.
    """

    def __init__(self, config_file="configs/default.json"):
        """
        The constructor for the BurstDatagen class.

        Args:
            config_file (str): The relative file path for the configuration file to be used for generating random burst definition objects.
        """
        with open(config_file, "r") as config_file_id:
            instance = json.load(config_file_id)
            self.config = instance
        self.rng = np.random.default_rng(self.config["generation"]["rand_seed"])

    def __check_collisions(self, candidate_burst, burst_list):
        """
        Checks if a burst overlaps with other bursts in time and frequency.

        Args:
            candidate_burst (obj): The burst_def objects, of type :class:`pywaspgen.burst_def.BurstDef`, to check for overlaps.
            burst_list (obj): The list of burst_def objects, of type :class:`pywaspgen.burst_def.BurstDef`, to check the candidate burst against.

        Returns:
            bool: True if ``candidate_burst`` overlaps in time and frequency with any burst in the ``burst_list``, otherwise False.
        """
        for burst in burst_list:
            if (burst.get_low_freq() >= candidate_burst.get_high_freq()) or (candidate_burst.get_low_freq() >= burst.get_high_freq()):
                continue
            if (burst.get_end_time() <= candidate_burst.start) or (candidate_burst.get_end_time() <= burst.start):
                continue
            return True
        return False

    def __check_not_observed(self, candidate_burst):
        """
        Checks if a burst is within the observation window.

        Args:
            candidate_burst (obj): The burst_def objects, of type :class:`pywaspgen.burst_def.BurstDef`, to check for overlaps.

        Returns:
            bool: True if ``candidate_burst`` is outside of the observation interval, otherwise False.
        """
        if (candidate_burst.start < 0 and candidate_burst.duration < np.abs(candidate_burst.start)) or candidate_burst.start > self.config["spectrum"]["observation_duration"]:
            return True
        return False

    def __get_random(self, rng, sig_type, parameter):
        """
        Generates a random parameter for the :class:`pywaspgen.burst_def.BurstDef` object.

        Args:
            rng (obj): A numpy random generator object used by the random generators.
            sig_type (dict): The signal type of the burst to generate the random burst ``parameter`` for.
            parameter (str):  The parameter of the :class:`pywaspgen.burst_def.BurstDef` object to get a random value for.

        Returns:
            int/float: The random value for the ``parameter`` of the :class:`pywaspgen.burst_def.BurstDef` object.
        """        
        override = 0
        if sig_type["label"] in self.config["sig_overrides"]:
            if "burst" in self.config["sig_overrides"][sig_type["label"]]:
                if parameter in self.config["sig_overrides"][sig_type["label"]]["burst"]:
                    override = 1
        if override == 1:
            range = self.config["sig_overrides"][sig_type["label"]]["burst"][parameter]
        else:
            range = self.config["sig_defaults"]["burst"][parameter]

        if type(range[0]).__name__ == "float":
            return rng.uniform(range[0], range[1])
        elif type(range[0]).__name__ == "int":
            return rng.integers(range[0], range[1], endpoint=True)

    def gen_burst(self, rng):
        """
        Generates a random :class:`pywaspgen.burst_def.BurstDef` object.

        Args:
            rng (obj): A numpy random generator object used by the random generators.

        Returns:
            obj: A :class:`pywaspgen.burst_def.BurstDef` object with random parameters in ranges specified by the configuration file.
        """
        sig_type = self.config["spectrum"]["sig_types"][rng.choice(len(self.config["spectrum"]["sig_types"]))]

        cent_freq = self.__get_random(rng, sig_type, "cent_freq")
        bandwidth = self.__get_random(rng, sig_type, "bandwidth")
        start = self.__get_random(rng, sig_type, "start")
        duration = self.__get_random(rng, sig_type, "duration")
                
        metadata = {"UUID": uuid.UUID(bytes=rng.bytes(16), version=4)}        
        return burst_def.BurstDef(cent_freq, bandwidth, start, duration, sig_type, metadata)

    def _gen_burstlist(self, rng):
        """
        Generates a random burst list of :class:`pywaspgen.burst_def.BurstDef` objects.

        Args:
            seed (int): Sets the seed for randomization.

        Returns:
            obj: A list of :class:`pywaspgen.burst_def.BurstDef` objects each with random parameters in ranges specified by the configuration file.
        """
        burst_list = []
        sig_count = 0
        stop_gen = False
        trial_count = 0
        while not stop_gen:
            candidate_burst = self.gen_burst(rng)
            if not self.__check_not_observed(candidate_burst) and (self.config["spectrum"]["allow_collisions_flag"] or (not self.config["spectrum"]["allow_collisions_flag"] and not self.__check_collisions(candidate_burst, burst_list))):
                burst_list.append(candidate_burst)
                sig_count += 1
                trial_count = 0
            else:
                trial_count += 1
            if trial_count == self.config["generation"]["max_attempts"]:
                stop_gen = True
            if sig_count == self.config["spectrum"]["max_signals"]:
                stop_gen = True
        return burst_list

    def gen_burstlist(self, batch_size=1):
        """
        Generates a batch of burst lists using multiprocessing across cpu cores.

        Args:
            batch_size (int): The number of burst lists to generate through multiprocessing.

        Returns:
            obj: A list, of size defined by ``batch_size``, of burst lists of burst_def objects, of type :class:`pywaspgen.burst_def.BurstDef`, with random parameters in ranges specified by the configuration file.
        """
        with multiprocessing.Pool(self.config["generation"]["pool"]) as pool:
            rngs = self.rng.spawn(batch_size)
            return list(pool.map(self._gen_burstlist, tqdm.tqdm(rngs, total=batch_size)))

    def plot_burstdata(self, burst_list, ax=[]):
        """
        Plots and labels the bounding boxes of all bursts.

        Args:
            burst_list (obj): The list of :class:`pywaspgen.burst_def.BurstDef` objects to plot.
            ax (obj): Matplotlib axis in which to put the plot.

        Returns:
            obj: Matplotlib labeled plot of the bounding boxes of all bursts in burst_list.
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
            plt.title("Spectrum Metadata Plot")
            ax.set_facecolor("xkcd:salmon")
        else:
            ax.set_title("Spectrogram River Plot with Overlaid Spectrum Metadata")

        color_list = [
            list(item)
            for item in distinctipy.get_colors(
                len(self.config["spectrum"]["sig_types"]),
                [(255, 121, 108)],
                rng=self.config["generation"]["rand_seed"],
            )
        ]

        sig_type_list = []
        if not isinstance(burst_list, list):
            burst_list = [burst_list]
        for burst in burst_list:
            sig_type_list.append(burst.sig_type["label"])
            patch = patches.Rectangle((burst.start, burst.get_low_freq()), burst.duration, burst.bandwidth, linewidth=3, linestyle="--", edgecolor=color_list[[k for k, d in enumerate(self.config["spectrum"]["sig_types"]) if burst.sig_type["label"] in d.values()][0]], fill=False)
            ax.add_patch(patch)

            rx, ry = patch.get_xy()
            cx = rx + patch.get_width() / 2.0
            cy = ry + patch.get_height() / 2.0
            ax.annotate(str(burst.metadata["UUID"])[0:5], (cx, cy), color=color_list[[k for k, d in enumerate(self.config["spectrum"]["sig_types"]) if burst.sig_type["label"] in d.values()][0]], weight="bold", fontsize=6, ha="center", va="center")

        sig_type_list = set(sig_type_list)
        legend_elements = []
        counter = 0
        for sig_type in sig_type_list:
            legend_elements.append(patches.Patch(facecolor=color_list[[k for k, d in enumerate(self.config["spectrum"]["sig_types"]) if sig_type in d.values()][0]], edgecolor="k", label=sig_type))
            counter += 1
        ax.legend(handles=legend_elements)
        ax.set_xlim([0, self.config["spectrum"]["observation_duration"]])
        return ax
