


import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.burst_def import BurstDef
from pywaspgen.iq_datagen import IQDatagen


class Pywaspgen:
    def __init__(self, config_file: str, output_directory: str, iterations=1):
        self.config_file = config_file
        self.output_directory = output_directory
        self.iterations = iterations

        self.iq_data = []
        self.burst_list = []
        self.updated_burst_list = []
        self.burst_gen = BurstDatagen(config_file)
        self.iq_gen = IQDatagen(config_file)

    def create_and_add_direct_burst_def(self, cent_freq: float, bandwidth: float, start: int, duration: int, sig_type: dict):
        """
        Creates a direst burst definition from passed in parameters

        Args:
            cent_freq (float): The center frequency of the burst
            bandwidth (float): The bandwidth of the burst
            start (int): The start time of the burst
            duration (int): The duration of the burst
            sig_type (dict): The type of signal of the burst
        """
        self.burst_list.append(
            BurstDef(
                cent_freq=cent_freq,
                bandwidth=bandwidth,
                start=start,
                duration=duration,
                sig_type=sig_type,
            )
        )

    def create_random_burst_def(self):
        """
        Creates a random burst definition
        """
        self.burst_list = self.burst_gen.gen_burstlist(self.iterations)
        return self.burst_list

    def generate_iq_data(self, batch_size=1, random=False, saving=False):
        """
        Generate IQ data from the current burst list
        """
        loop_count = self.iterations // batch_size
        prev_start = 0
        for i in range(loop_count):
            if random:
                burst_list = self.burst_list[prev_start : prev_start + batch_size]
            else:
                burst_list = [self.burst_list][prev_start : prev_start + batch_size]
            self.iq_data, self.updated_burst_list = self.iq_gen.gen_iqdata(burst_list)
            if saving:
                print(f"Saving off this batch of IQ : {i} of {loop_count} \n")
                for k in range(len(self.updated_burst_list)):
                    self.save_data(burst_entry=self.updated_burst_list[k], iq_data=self.iq_data[k])
            prev_start = prev_start + batch_size


    def plot_burst_data(self, random=False, updated=False):
        """
        Plots the burst data

        Args:
            random (bool, optional): True if the burst list was randomly generated. Defaults to False.
            updated (bool, optional): True if the updated_burst_list should be plot as opposed
                to the original burst_list. Defaults to False.
        """
        if updated:
            self.burst_gen.plot_burstdata(self.updated_burst_list[0])
        elif random:
            self.burst_gen.plot_burstdata(self.burst_list[0])
        else:
            self.burst_gen.plot_burstdata(self.burst_list)
        plt.show()

    def plot_iq_data(self, with_burst=True):
        """
        Plots the current IQ data
        """
        if with_burst:
            ax = self.burst_gen.plot_burstdata(self.updated_burst_list[0])
            self.iq_gen.plot_iqdata(self.iq_data[0], ax)
        else:
            self.iq_gen.plot_iqdata(self.iq_data[0])
        plt.show()

    def save_data(self, burst_entry, iq_data):
        """
        Saves the IQ data for each burst entry
        """
        for b in burst_entry:
            corrected_iq_data = iq_data[b.get_start_time() : b.get_end_time()]  # Corrects the sample offset of the received samples.
            corrected_iq_data = impairments.freq_off(corrected_iq_data, -b.get_center_freq())  # Corrects the frequency offset of the received samples.
            # Save IQ
            unique_id = str(b.get_meta_uuid())
            np.save(self.output_directory + "/iq/" + unique_id, corrected_iq_data)
            with open(self.output_directory + "/meta/" + unique_id + ".pkl", "wb") as f:
                pickle.dump(b, f)