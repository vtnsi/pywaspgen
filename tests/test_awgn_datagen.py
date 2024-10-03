"""
Unit Test for Verification of Modem against Theoretical in an AWGN channel.
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import unittest

import numpy as np

from scripts.awgn_datagen_check import AWGNDataGenCheck


class bust_datagen_tester(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_snr_theoretical_match(self):
        snr_db_range = (0, 20)
        hndl_ = AWGNDataGenCheck()
        ser_sim, ser_theory = hndl_.generate_theory_and_sim_results(snr_db_range=snr_db_range)

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
