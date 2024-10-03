"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import unittest

import numpy as np

from scripts.awgn_digital_modem_check import AWGNDigitalModemCheck


class bust_datagen_tester(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_snr_theoretical_match(self):
        snr_db_range = (0, 20)
        num_symbols = 100000

        sig_type = {"format": "qam", "order": 64}
        pulse_type = {
            "sps": 5.2,
            "format": "RRC",
            "params": {"beta": 0.4, "span": 10, "window": ("kaiser", 5.0)},
        }

        hndl_ = AWGNDigitalModemCheck(sig_type=sig_type, pulse_type=pulse_type)
        ser_sim, ser_theory = hndl_.generate_theory_and_sim_results(snr_db_range=snr_db_range, num_symbols=num_symbols)
        # How Close do we expect this to be
        self.assertEqual(True, np.allclose(ser_sim, ser_theory, 0.1))


if __name__ == "__main__":
    unittest.main()
