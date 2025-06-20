"""
A test script for checking if a given digital modem has been implemented correctly by checking its simulated performance against its theoretical performance in an Additive White Guassian Noise (AWGN) channel.
"""
import matplotlib.pyplot as plt
import tqdm

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

# Instantiate a burst generator object and specify the signal parameter configuration file to use.
burst_gen = BurstDatagen("configs/awgn_check.json")
# Instantiate an iq generator object and specify the signal parameter configuration file to use.
iq_gen = IQDatagen("configs/awgn_check.json")

snr_db_range = (0,8)      # The range of signal-to-noise ratios (SNRs) to calculate the performance of.

ser_sim = []
ser_theory = []
for snr_db in tqdm.trange(snr_db_range[0], snr_db_range[1] + 1):
    iq_gen.config["sig_defaults"]["iq"]["snr"] = (snr_db, snr_db)   # Overwrites the configuration file to generate a specific SNR.

    rand_burst = burst_gen.gen_burstlist()                          # Use the instantiated burst generator object to create a random signal burst based on the chosen configuration file.

    iq_data, updated_rand_burst = iq_gen.gen_iqdata(rand_burst)     # Based on the random signal burst, generates received samples impacted by an AWGN channel.

    corrected_iq_data = iq_data[0][updated_rand_burst[0][0].start:updated_rand_burst[0][0].start+updated_rand_burst[0][0].duration]             # Corrects the sample offset of the received samples.
    corrected_iq_data = impairments.freq_off(corrected_iq_data, -updated_rand_burst[0][0].cent_freq)                                   # Corrects the frequency offset of the received samples.

    # Calculates the number of symbols different between the transmit symbols and the received symbols and then calculates the symbol error rate (SER).
    ser_sim.append(updated_rand_burst[0][0].metadata["modem"].get_sim_awgn(corrected_iq_data))
    ser_theory.append(updated_rand_burst[0][0].metadata["modem"].get_theory_awgn(snr_db))      # Generates the theoretical SER (theoretical equation assumed to be provided by the modem itself).

# Plotting SER Simulation and Theory Curves
plt.figure(figsize=(15, 5))
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.legend()
plt.show()
