import numpy as np
from scipy.special import comb
from pywaspgen.modem.digital.digital import DIGITAL

class FSK(DIGITAL):
    def __init__(self, outer_deviation, symbol_rate, sig_type={"format": "fsk", "order": 2}):
        self.outer_deviation = outer_deviation
        self.symbol_rate = symbol_rate        
        self.sig_type = sig_type  
        super().__init__(sig_type)      
         
    def _gen_samples(self, num_samples):
        total_symbols = int(num_samples * self.symbol_rate)
        if total_symbols >= 1:
            symbols = np.random.choice(self.symbol_table, total_symbols)
            return self.get_samples(symbols)
        else:
            return np.array([])        

    def _get_symbols(self, samples):
        symbols = []
        num_symbols = int(len(samples)*self.symbol_rate)
        t = list(range(num_symbols*int(1/self.symbol_rate)))
        for k in range(num_symbols):
            powers = []
            times = t[k*int(1/self.symbol_rate):(k+1)*int(1/self.symbol_rate)]
            for freq in self.symbol_table:
                times = t[k*int(1/self.symbol_rate):(k+1)*int(1/self.symbol_rate)]
                powers.append(np.sum(samples[k*int(1/self.symbol_rate):(k+1)*int(1/self.symbol_rate)] * [np.exp(-2.0j * np.pi * freq * x) for x in times])**2.0)

            symbols.append(self.symbol_table[np.argmax(powers)])
        return symbols    
            
    def _get_theory_awgn(self, snr_db):
        snr_lin = 10.0 ** (snr_db / 10.0)
        if self.sig_type["format"] == "fsk":
            prob_error = 0
            for k in range(1, self.order):
                prob_error += (-1)**(k+1)*comb(self.order-1,k)*(1.0/(k+1))*np.exp(-np.log2(self.order)*((k/(k+1))*snr_lin))
            return prob_error
        
    def _symbol_table_create(self):
        deviation = self.outer_deviation / (self.sig_type["order"]-1)
        for k in range(1, int(self.sig_type["order"] / 2) + 1):
            self.symbol_table.append((-2.0 * k + 1.0)*deviation)
            self.symbol_table.append((2.0 * k - 1.0)*deviation)        