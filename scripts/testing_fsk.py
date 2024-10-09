import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import quad

def integrand(y, order, snr_lin):
    return ((1.0-(1.0-(0.5*erfc(y/np.sqrt(2))))**(order-1))*np.exp(-0.5*(y-np.sqrt(2*np.log2(order)*snr_lin))**2.0))/np.sqrt(2*np.pi)

order = 4
num_symbols = 10000
BW = np.random.uniform(0.1, 0.5)
H = np.random.uniform(1.0, 3.0)

Rs = BW/(H*((order-1)**2.0)+2.0)
deviation = (H*Rs*(order-1))/2.0

samples_per_symbol = int(1/Rs)
Rs = 1/samples_per_symbol
deviation = (H*Rs*(order-1))/2.0

snr_db_range = (0, 10)

BW_est = 2.0*(order-1)*deviation + 2.0*Rs
H_est = (2.0*deviation)/(Rs*(order-1))

symbol_table = []
for k in range(1, int(order / 2) + 1):
    symbol_table.append((-2.0 * k + 1.0)*deviation)
    symbol_table.append((2.0 * k - 1.0)*deviation)

ser_sim = []
ser_theory = []
for snr_db in tqdm.trange(snr_db_range[0], snr_db_range[1] + 1):
    snr_lin = 10.0 ** (snr_db / 10.0)

    ## Transmitter
    tx_symbols = np.random.choice(list(range(order)), num_symbols)

    t = list(range(len(tx_symbols)*samples_per_symbol))
    tx_signal = []
    for k in range(len(tx_symbols)):
        tx_signal = np.concatenate((tx_signal, [np.exp(2.0j * np.pi * (symbol_table[tx_symbols[k]]) * x) for x in t[k*samples_per_symbol:(k+1)*samples_per_symbol]]))
    tx_signal = tx_signal/np.sqrt(samples_per_symbol/(2.0*np.log2(order)))

    ## Channel
    noise_samps = np.random.normal(0.0, np.sqrt(0.5), len(tx_signal)) + 1.0j * (np.random.normal(0.0, np.sqrt(0.5), len(tx_signal)))
    rx_signal = np.sqrt(snr_lin / 2.0) * tx_signal + noise_samps

    ## Receiver
    rx_symbols = np.zeros(num_symbols)
    for k in range(num_symbols):
        metric = np.zeros(order)
        for kk in range(order):
            metric[kk] = np.dot(rx_signal[k*samples_per_symbol:(k+1)*samples_per_symbol],[np.exp(-2.0j * np.pi * (symbol_table[kk]) * x) for x in t[k*samples_per_symbol:(k+1)*samples_per_symbol]])
        rx_symbols[k] = np.argmax(metric)

    ## Simulation Error Rate Calc
    symbol_errors = 0
    for k in range(num_symbols):
        if not tx_symbols[k] == rx_symbols[k]:
            symbol_errors += 1
    ser_sim.append(symbol_errors/num_symbols)

    ## Theoretical Error Rate Calc
    ser_theory.append(quad(integrand, -np.inf, np.inf, args=(order, snr_lin))[0])
                      
# Plotting SER Simulation and Theory Curves
print(ser_sim)
print(ser_theory)

plt.figure(figsize=(15, 5))
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_sim, "b-*", label="Simulated")
plt.semilogy(range(snr_db_range[0], snr_db_range[1] + 1), ser_theory, "r-*", label="Theory")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.title("Symbol Error Rate for " + str(order) + "-FSK")
plt.legend()
plt.grid()
plt.show()