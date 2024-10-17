import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
from pywaspgen.modem.digital.digital import DIGITAL

class FSK(DIGITAL):
    def __init__(self, deviation, symbol_rate, samples_per_symbol, sig_type):
        self.deviation = deviation
        self.symbol_rate = symbol_rate   
        self.samples_per_symbol = samples_per_symbol      
        super().__init__(sig_type)      
         
    def _gen_samples(self, num_samples):
        total_symbols = int(num_samples * self.symbol_rate)+1
        if total_symbols >= 1:
            symbols = self.gen_symbols(total_symbols)      
            t = list(range(len(symbols)*self.samples_per_symbol))
            samples = []
            for k in range(len(symbols)):
                samples = np.concatenate((samples, [np.exp(2.0j * np.pi * symbols[k] * x) for x in t[k*self.samples_per_symbol:(k+1)*self.samples_per_symbol]]))
            return (samples/np.sqrt(self.samples_per_symbol/(2.0*np.log2(self.order))))[0:num_samples]
        else:
            return np.array([])  
              
    def integrand(self, y, snr_lin):
        return ((1.0-(1.0-(0.5*erfc(y/np.sqrt(2))))**(self.order-1))*np.exp(-0.5*(y-np.sqrt(2*np.log2(self.order)*snr_lin))**2.0))/np.sqrt(2*np.pi)

    def _get_symbols(self, samples):
        t = list(range(len(samples)))
        demoded_symbols = np.zeros(len(self.generated_symbols))
        for k in range(len(self.generated_symbols)):
            metric = np.zeros(self.order)
            for kk in range(self.order):
                metric[kk] = np.vdot([np.exp(2.0j * np.pi * (self.symbol_table[kk]) * x) for x in t[k*self.samples_per_symbol:(k+1)*self.samples_per_symbol]], samples[k*self.samples_per_symbol:(k+1)*self.samples_per_symbol])
            demoded_symbols[k] = np.argmax(metric)
        return demoded_symbols    
            
    def _get_theory_awgn(self, snr_lin):
        return quad(self.integrand, -np.inf, np.inf, args=(snr_lin))[0]
        
    def _symbol_table_create(self):
        for k in range(1, int(self.sig_type["order"] / 2) + 1):
            self.symbol_table.append((-2.0 * k + 1.0)*self.deviation)
            self.symbol_table.append((2.0 * k - 1.0)*self.deviation)        