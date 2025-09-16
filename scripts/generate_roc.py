import csv
import numpy as np
import matplotlib.pyplot as plt
import pywaspgen

if __name__ == "__main__":
    num_channels = 8
    num_sense_samps = 10000

    num_trials = 10000
    snr_range_db = (-20, 10)

    chan_cent_freqs = np.linspace(-0.5+1.0/(2.0*num_channels), 0.5-1.0/(2.0*num_channels), num=num_channels, endpoint=True)
    num_bins_per_chan = int(num_sense_samps/num_channels)       # For simplicity make sure num_sense_samps/num_channels is an int.

    iq_gen = pywaspgen.IQDatagen("configs/default.json")
    iq_gen.config['spectrum']['observation_duration'] = num_sense_samps

    snrs_db = np.arange(snr_range_db[0], snr_range_db[1]+1, 1)
    thresholds = np.linspace(0.0-2.0, 3, num=1000)

    pf_interp = np.linspace(0.0, 1.0, num=101, endpoint=True)
    pd_interp = np.zeros((len(snrs_db), len(pf_interp)), dtype=float)
    avg_rx_pows = np.zeros(len(snrs_db), dtype=float)
    for snr_idx in range(len(snrs_db)):
        print(snrs_db[snr_idx])
        burst_list = []

        # Signal Present Tests
        soi_chan = np.random.choice(chan_cent_freqs, num_trials)
        pd = np.zeros(len(thresholds), dtype=float)
        pf = np.zeros(len(thresholds), dtype=float)
        for trial in range(num_trials):      
            burst_list.append([pywaspgen.BurstDef(cent_freq=soi_chan[trial], bandwidth=(1.0/num_channels), start=0, duration=num_sense_samps, sig_type={"type": "psk", "order": 4, "label": "QPSK"})])   
        iq_gen.config['sig_defaults']['iq']['snr'] = [snrs_db[snr_idx], snrs_db[snr_idx]]
        iq_data, updated_burst = iq_gen.gen_iqdata(burst_list)

        for trial in range(num_trials):            
            fft_data = np.fft.fftshift(np.fft.fft(iq_data[trial], norm='ortho'))
            chan_pows = np.zeros(num_channels, dtype=float)
            for chan in range(num_channels):
                chan_pows[chan] = 10.0*np.log10(np.sum(np.abs(fft_data[(chan*num_bins_per_chan):((chan+1)*num_bins_per_chan)])**2.0)/(num_bins_per_chan))
            avg_rx_pows[snr_idx] += chan_pows[np.where(chan_cent_freqs==soi_chan[trial])[0][0]]/num_trials 

            for threshold_idx in range(len(thresholds)):
                detects = chan_pows > thresholds[threshold_idx]
                if any(detects):
                    pd[threshold_idx] += 1.0/num_trials

        # Signal Absent Tests
        for trial in range(num_trials):      
            iq_data = pywaspgen.impairments.awgn(np.zeros(num_sense_samps, dtype=np.csingle), -np.inf)
            fft_data = np.fft.fftshift(np.fft.fft(iq_data, norm='ortho'))

            for chan in range(num_channels):
                chan_pows[chan] = 10.0*np.log10(np.sum(np.abs(fft_data[(chan*num_bins_per_chan):((chan+1)*num_bins_per_chan)])**2.0)/(num_bins_per_chan))

            for threshold_idx in range(len(thresholds)):
                detects = chan_pows > thresholds[threshold_idx]
                if any(detects):
                    pf[threshold_idx] += 1.0/num_trials

        # Interpolate PDs to specific PF values.
        pf_unique, idx = np.unique(pf, return_index=True)
        pd_unique = pd[idx]
        pd_interp[snr_idx][:] = np.interp(pf_interp, pf_unique, pd_unique)

    csv_file_name = str(num_sense_samps) + '_roc_table_interpolated.csv'
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR', 'Average Power'] + [f"{x:.2f}" for x in pf_interp])
        for snr_idx in range(len(snrs_db)):
            writer.writerow([snrs_db[snr_idx], f"{avg_rx_pows[snr_idx]:.3f}"] + [f"{x:.2f}" for x in pd_interp[snr_idx][:]])

    for snr_idx in range(len(snrs_db)):
        plt.plot(pf_interp[:], pd_interp[snr_idx][:], '-*', label='SNR (dB) = ' + str(snrs_db[snr_idx]))
    plt.title('ROC Curve for $N_{samps}$ = ' + str(num_sense_samps) + ' and $N_{chans}$ = ' + str(num_channels))
    plt.legend()
    plt.show()