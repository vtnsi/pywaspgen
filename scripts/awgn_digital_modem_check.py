"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""

import pywaspgen.impairments as impairments
import pywaspgen.modem as modem
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # In PyWASPGEN, a modem defines the type of communication format used to generate in-phase/quadrature (IQ) data and is typically defined by its 'sig_type' and its 'pulse_type'.
    #   sig_type - Specifies for the given modem the sub-class parameters of the communication format.
    #   pulse_type - Specifies for the given modem the pulse shape parameters used to shape the spectral content of the IQ data.
    sig_type = {"format": "fsk", "order": 2}
    pulse_type = {"sps": 2.3, "format": "RRC", "params": {"beta": 0.4, "span": 10, "window": ("kaiser", 5.0)}}
    
    # Instantiate the modem object and specifies the signal type and pulse shaping parameters.
    modem = modem.ask(sig_type, pulse_type)

    num_symbols = 100000  # The number of data symbols to be used to calculate the simulated performance. Higher number will approach theory better but take longer to execute.
    snr_db_range = (0, 20)  # The range of signal-to-noise ratios (SNRs) to calculate the performance of.

    ser_sim = []
    ser_theory = []
    for snr_db in range(snr_db_range[0], snr_db_range[1] + 1):
        tx_samples = modem.gen_samples(num_symbols)  # Modulates the digital data symbols to get the transmit samples.

        rx_samples = impairments.awgn(tx_samples, snr_db)  # Applies the AWGN channel to the transmit samples to get the received samples.

        # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
        ser_sim.append(modem.get_sim_awgn(rx_samples))
        ser_theory.append(modem.get_theory_awgn(snr_db))  # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).

    # Plotting SER Simulation and Theory Curves
    plt.figure(figsize=(15, 5))
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
    plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.title("Signal Type: " + str(sig_type) + "\nPulse Type: " + str(pulse_type))
    plt.legend()
    plt.show()