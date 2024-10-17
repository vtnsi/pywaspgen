"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import pywaspgen.impairments as impairments
import pywaspgen.modems as modems


class AWGNDigitalModemCheck:
    def __init__(self, sig_type: dict = {"format": "pam", "order": 64}, pulse_type: dict = {"sps": 5.2, "format": "RRC", "params": {"beta": 0.4, "span": 10, "window": ("kaiser", 5.0)}}):
        # In PyWASPGEN, a modem defines the type of communication format used to generate in-phase/quadrature (IQ) data and is typically defined by its 'sig_type' and its 'pulse_type'.
        #   sig_type - Specifies for the given modem the sub-class parameters of the communication format.
        #   pulse_type - Specifies for the given modem the pulse shape parameters used to shape the spectral content of the IQ data.
        self.sig_type = sig_type
        self.pulse_type = pulse_type

        # Instantiate the modem object and specifies the signal type and pulse shaping parameters.
        self.modem = modems.LDAPM(self.sig_type, self.pulse_type)

    def generate_theory_and_sim_results(self, snr_db_range: tuple = (0, 20), num_symbols: int = 10000):
        """
        Args:
            snr_db_range : tuple The range of signal-to-noise ratios (SNRs) to calculate the performance of.
            num_symbols : int The number of data symbols to be used to calculate the simulated performance. Higher number will approach theory better but take longer to execute.
        Returns:
            np arrays - Simulated, and Theory curve
        """
        ser_sim = []
        ser_theory = []
        for snr_db in tqdm.trange(snr_db_range[0], snr_db_range[1] + 1):
            tx_symbols = self.modem.gen_symbols(num_symbols, snr_db)  # Generates the digital data symbols to transmit.
            tx_samples = self.modem.get_samples(tx_symbols)  # Modulates the digital data symbols to get the transmit samples.

            rx_samples = impairments.awgn(tx_samples)  # Applies the AWGN channel to the transmit samples to get the received samples.
            rx_symbols = self.modem.get_symbols(rx_samples)  # Demodulates the received samples to get the received symbols.

            # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
            symbol_errors = 0
            for k in range(0, len(rx_symbols)):
                symbol_errors = symbol_errors + (1 - np.equal(tx_symbols[k], self.modem.get_nearest_symbol(rx_symbols[k], snr_db)))
            ser_sim.append(symbol_errors / len(rx_symbols))

            ser_theory.append(self.modem.get_theory_awgn(snr_db))  # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).

        return np.array(ser_sim), np.array(ser_theory)


if __name__ == "__main__":
    snr_db_range = (0, 8)
    num_symbols = 100000

    sig_type = {"format": "qam", "order": 16}
    pulse_type = {"sps": 2.7, "format": "RRC", "params": {"beta": 0.35, "span": 10, "window": ("kaiser", 5.0)}}

    hndl_ = AWGNDigitalModemCheck(sig_type=sig_type, pulse_type=pulse_type)
    ser_sim, ser_theory = hndl_.generate_theory_and_sim_results(snr_db_range=snr_db_range, num_symbols=num_symbols)

    # Plotting SER Simulation and Theory Curves
    plt.figure(figsize=(15, 5))
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.title("Signal Type: " + str(sig_type) + "\nPulse Type: " + str(pulse_type))
    plt.legend()
    plt.show()
