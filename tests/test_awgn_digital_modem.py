"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import pywaspgen.impairments as impairments
import pywaspgen.modems as modems

import unittest
class bust_datagen_tester(unittest.TestCase):
    
    def setUp(self):

        # In PyWASPGEN, a modem defines the type of communication format used to generate in-phase/quadrature (IQ) data and is typically defined by its 'sig_type' and its 'pulse_type'.
        #   sig_type - Specifies for the given modem the sub-class parameters of the communication format.
        #   pulse_type - Specifies for the given modem the pulse shape parameters used to shape the spectral content of the IQ data.
        self.sig_type = {"format": "qam", 
                    "order": 64}
        self.pulse_type = {
            "sps": 5.2,
            "format": "RRC",
            "params": {"beta": 0.4, "span": 10, "window": ("kaiser", 5.0)},
        }
        # Instantiate the modem object and specifies the signal type and pulse shaping parameters.
        self.modem = modems.LDAPM(self.sig_type, self.pulse_type)

        self.num_symbols = 100000            # The number of data symbols to be used to calculate the simulated performance. Higher number will approach theory better but take longer to execute.
        self.snr_db_range = (0, 20)          # The range of signal-to-noise ratios (SNRs) to calculate the performance of.
        
    def test_symbol_error(self):
        ser_sim = []
        ser_theory = []
        for snr_db in tqdm.trange(self.snr_db_range[0], self.snr_db_range[1] + 1):
            tx_symbols = self.modem.gen_symbols(self.num_symbols)         # Generates the digital data symbols to transmit.
            tx_samples = self.modem.get_samples(tx_symbols)          # Modulates the digital data symbols to get the transmit samples.

            rx_samples = impairments.awgn(tx_samples, snr_db)   # Applies the AWGN channel to the transmit samples to get the received samples.
            rx_symbols = self.modem.get_symbols(rx_samples)          # Demodulates the received samples to get the received symbols.

            # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
            symbol_errors = 0
            for k in range(0, len(rx_symbols)):
                symbol_errors = symbol_errors + (1 - np.equal(tx_symbols[k], self.modem.get_nearest_symbol(rx_symbols[k])))
            ser_sim.append(symbol_errors / len(rx_symbols))

            ser_theory.append(self.modem.get_theory_awgn(snr_db))    # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).
            
        # How Close do we expect this to be
        self.assertEqual(True,np.allclose(ser_sim,ser_theory,0.1))
        
    
if __name__ == '__main__':
    unittest.main()
