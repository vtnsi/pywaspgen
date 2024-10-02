"""
Unit Test for Verification of Modem against Theoretical in an AWGN channel.
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import unittest

import numpy as np
import tqdm

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen


class bust_datagen_tester(unittest.TestCase):
    def setUp(self):
        # Instantiate a burst generator object and specify the signal parameter configuration file to use.
        self.burst_gen_hndl = BurstDatagen("configs/awgn_check.json")

        # Instantiate an iq generator object and specify the signal parameter configuration file to use.
        self.iq_gen_hndl = IQDatagen("configs/awgn_check.json")

        # SNR RANGE
        self.snr_db_range = (0, 20)  # The range of signal-to-noise ratios (SNRs) to calculate the performance of.

    def tearDown(self):
        pass

    def test_snr_theoretical_match(self):
        ser_sim = []
        ser_theory = []
        for snr_db in tqdm.trange(self.snr_db_range[0], self.snr_db_range[1] + 1):
            self.iq_gen_hndl.config["iq_defaults"]["snr"] = (snr_db, snr_db)  # Overwrites the configuration file to generate a specific SNR.

            rand_burst = self.burst_gen_hndl.gen_burstlist()  # Use the instantiated burst generator object to create a random signal burst based on the chosen configuration file.

            iq_data, updated_rand_burst = self.iq_gen_hndl.gen_iqdata(rand_burst)  # Based on the random signal burst, generates received samples impacted by an AWGN channel.

            corrected_iq_data = iq_data[updated_rand_burst[0].start : updated_rand_burst[0].duration]  # Corrects the sample offset of the received samples.
            corrected_iq_data = impairments.freq_off(corrected_iq_data, -updated_rand_burst[0].cent_freq)  # Corrects the frequency offset of the received samples.

            tx_symbols = updated_rand_burst[0].metadata["modem"].generated_symbols  # Gets the digital data symbols that were transmitted.
            rx_symbols = updated_rand_burst[0].metadata["modem"].get_symbols(corrected_iq_data)  # Demodulates the corrected received samples to get the received symbols.

            # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
            symbol_errors = 0
            for k in range(0, len(rx_symbols)):
                symbol_errors = symbol_errors + (1 - np.equal(tx_symbols[k], updated_rand_burst[0].metadata["modem"].get_nearest_symbol(rx_symbols[k])))
            ser_sim.append(symbol_errors / len(rx_symbols))
            ser_theory.append(updated_rand_burst[0].metadata["modem"].get_theory_awgn(snr_db))  # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).

        # Assert that we're all close to within 0.1 of the theoretical value
        self.assertEqual(np.allclose(ser_sim, ser_theory, rtol=0.1), True)

    def test_other_configs(self):
        # TODO: As we change signal types or configurations
        # Add the tests here or better yet - Just loop through all configurations matching a certain
        # glob pattern
        # Add other tests for specific verification of corner cases and generation.
        pass


if __name__ == "__main__":
    unittest.main()
