"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen


class AWGNDataGenCheck:
    def __init__(self, burst_gen_config="configs/awgn_check.json", iq_gen_config="configs/awgn_check.json"):
        # Instantiate a burst generator object and specify the signal parameter configuration file to use.
        self.burst_gen = BurstDatagen(burst_gen_config)
        # Instantiate an iq generator object and specify the signal parameter configuration file to use.
        self.iq_gen = IQDatagen(iq_gen_config)

    def generate_theory_and_sim_results(self, snr_db_range: tuple = (0, 20)) -> tuple[np.array, np.array]:
        """
        Generates the Theory and Simulated Curve set for the defined snr range.

        Args:
            snr db range (tuple): Range in dB to run the simulation and theory

        Returns:
            np arrays - Simulated, and Theory curve
        """
        ser_sim = []
        ser_theory = []

        for snr_db in tqdm.trange(snr_db_range[0], snr_db_range[1] + 1):
            self.iq_gen.config["iq_defaults"]["snr"] = (snr_db, snr_db)  # Overwrites the configuration file to generate a specific SNR.

            rand_burst = self.burst_gen.gen_burstlist()  # Use the instantiated burst generator object to create a random signal burst based on the chosen configuration file.

            iq_data, updated_rand_burst = self.iq_gen.gen_iqdata(rand_burst)  # Based on the random signal burst, generates received samples impacted by an AWGN channel.

            corrected_iq_data = iq_data[updated_rand_burst[0].start : updated_rand_burst[0].start + updated_rand_burst[0].duration]  # Corrects the sample offset of the received samples.
            corrected_iq_data = impairments.freq_off(corrected_iq_data, -updated_rand_burst[0].cent_freq)  # Corrects the frequency offset of the received samples.

            tx_symbols = updated_rand_burst[0].metadata["modem"].generated_symbols  # Gets the digital data symbols that were transmitted.
            rx_symbols = updated_rand_burst[0].metadata["modem"].get_symbols(corrected_iq_data)  # Demodulates the corrected received samples to get the received symbols.

            # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
            symbol_errors = 0
            for k in range(0, len(rx_symbols)):
                symbol_errors = symbol_errors + (1 - np.equal(tx_symbols[k], updated_rand_burst[0].metadata["modem"].get_nearest_symbol(rx_symbols[k], snr_db)))
            ser_sim.append(symbol_errors / len(rx_symbols))

            ser_theory.append(updated_rand_burst[0].metadata["modem"].get_theory_awgn(snr_db))  # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).

        return np.array(ser_sim), np.array(ser_theory)


if __name__ == "__main__":
    awgn_hndl = AWGNDataGenCheck()
    snr_db_range = (0, 20)
    ser_sim, ser_theory = awgn_hndl.generate_theory_and_sim_results(snr_db_range=snr_db_range)

    # Plotting SER Simulation and Theory Curves
    plt.figure(figsize=(15, 5))
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.legend()
    plt.show()
