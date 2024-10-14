class FSK:
    def __init__(
        self,
        outer_deviation,
        symbol_rate,        
        sig_type={"format": "fsk", "order": 2},
    ):
        self.outer_deviation = outer_deviation
        self.symbol_rate = symbol_rate        
        self.sig_type = sig_type        
        self.__symbol_table_create()

    def __symbol_table_create(self):
        self.symbol_table = []
        deviation = self.outer_deviation / (self.sig_type["order"]-1)
        for k in range(1, int(self.sig_type["order"] / 2) + 1):
            self.symbol_table.append((-2.0 * k + 1.0)*deviation)
            self.symbol_table.append((2.0 * k - 1.0)*deviation)

    def gen_symbols(self, num_symbols):
        self.generated_symbols = np.random.choice(self.symbol_table, num_symbols)
        return self.generated_symbols

    def get_samples(self, symbols):
        t = list(range(len(symbols)*int(1/self.symbol_rate)))
        samples = []
        for k in range(len(symbols)):
            times = t[k*int(1/self.symbol_rate):(k+1)*int(1/self.symbol_rate)]
            samples = np.concatenate((samples, [np.exp(2.0j * np.pi * (symbols[k]) * x) for x in times]))
        return samples/np.sqrt((1/self.symbol_rate)/self.sig_type["order"])
         
    def gen_samples(self, num_samples):
        total_symbols = int(num_samples * self.symbol_rate)
        if total_symbols >= 1:
            symbols = np.random.choice(self.symbol_table, total_symbols)
            return self.get_samples(symbols)
        else:
            return np.array([])        

    def get_symbols(self, samples):
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
            
    def get_theory_awgn(self, snr_db):
        M = self.sig_type["order"]
        snr_lin = 10.0 ** (snr_db / 10.0)
        if self.sig_type["format"] == "fsk":
            prob_error = 0
            for k in range(1, M):
                prob_error += (-1)**(k+1)*comb(M-1,k)*(1.0/(k+1))*np.exp(-np.log2(M)*((k/(k+1))*snr_lin))
            return prob_error