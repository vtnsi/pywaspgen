import scipy.special
import tqdm
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pywaspgen.impairments as impairments

order = 2
num_symbols = 10000
BW = np.random.uniform(0.1, 0.5)
H = np.random.uniform(0.5, 3.0)

Rs = BW/(H*((order-1)**2.0)+2.0)
deviation = (H*Rs*(order-1))/2.0

samples_per_symbol = int(1/Rs)
snr_db_range = (0, 10)

BW_est = 2.0*(order-1)*deviation + 2.0*Rs
H_est = (2.0*deviation)/(Rs*(order-1))

freq = deviation
# shift = |F2-F1|
# deviation = shift/2
# cent_freq = (F2+F1)/2

ser_sim = []
ser_theory = []
for snr_db in tqdm.trange(snr_db_range[0], snr_db_range[1] + 1):
    ## Transmitter
    tx_symbols = np.random.choice([0,1], num_symbols)
    t = list(range(len(tx_symbols)*samples_per_symbol))
    tx_signal = []
    for k in range(len(tx_symbols)):
        times = t[k*samples_per_symbol:(k+1)*samples_per_symbol]
        if tx_symbols[k] == 0:
            tx_signal = np.concatenate((tx_signal, [np.exp(2.0j * np.pi * (-freq) * x) for x in times]))
        elif tx_symbols[k] == 1:
            tx_signal = np.concatenate((tx_signal, [np.exp(2.0j * np.pi * (freq) * x) for x in times]))
    tx_signal = tx_signal/np.sqrt(samples_per_symbol/2.0)

    ## Channel
    rx_signal = impairments.awgn(tx_signal, snr_db)
    
    #plt.psd(rx_signal, 2048, Fs=1.0)
    #plt.show()
    #exit()

    ## Receiver
    rx_symbols = []
    for k in range(num_symbols):
        times = t[k*samples_per_symbol:(k+1)*samples_per_symbol]
        power_0 = np.sum(rx_signal[k*samples_per_symbol:(k+1)*samples_per_symbol] * [np.exp(-2.0j * np.pi * (-freq) * x) for x in times])**2.0
        power_1 = np.sum(rx_signal[k*samples_per_symbol:(k+1)*samples_per_symbol] * [np.exp(-2.0j * np.pi * (freq) * x) for x in times])**2.0

        if power_0 > power_1:
            rx_symbols.append(0)
        else:
            rx_symbols.append(1)

    symbol_errors = 0
    for k in range(0,num_symbols):
        if not tx_symbols[k] == rx_symbols[k]:
            symbol_errors += 1
    ser_sim.append(symbol_errors/num_symbols)

    bino = scipy.special.comb(order-1,1)
    snr_lin = 10**(snr_db/10.0)
    ser_theory.append((-1)**(2.0)*bino*(1.0/(2.0))*np.exp(-(np.log2(2)*(1.0/2.0)*snr_lin)))

print(ser_sim)
print(ser_theory)

# Plotting SER Simulation and Theory Curves
plt.figure(figsize=(15, 5))
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.legend()
plt.grid()
plt.show()