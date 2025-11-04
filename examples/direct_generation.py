"""
An example script for showing how to directly generate burst definition objects for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import pywaspgen
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Instantiate a burst generator object (note: here only to be used for plotting the burst list).
    burst_gen = pywaspgen.BurstDatagen()

    # Instantiate an iq generator object and specify the signal parameter configuration file to use (note: directly setting iq specific signal parameters will be added in a future update).
    iq_gen = pywaspgen.IQDatagen("configs/default.json")

    # In PyWASPGEN, a burst defines the 'extent' and 'type' of signal to be created.
    #   extent - The time/frequency bounds of the signal to be generated.
    #   type - The signal format and associated parameters of the signal to be generated.
    # Here, we create a list of three user-defined signal bursts (for a complete list of the signal parameters and their ranges, see doc/parameters.txt).
    user_burst_list = [
        pywaspgen.BurstDef(
            cent_freq=0.3,
            bandwidth=0.2,
            start=1000,
            duration=50000,
            sig_type={"type": "ask", "order": 2, "label": "BASK"},
        )
    ]
    user_burst_list.append(
        pywaspgen.BurstDef(
            cent_freq=-0.3,
            bandwidth=0.1,
            start=5000,
            duration=8000,
            sig_type={"type": "qam", "order": 16, "label": "16QAM"},
        )
    )

    # Print and plot the list of user-defined signal bursts (note that each burst is given a universally unique identifier).
    print("\nUser-Defined Burst List:")
    print(user_burst_list)
    burst_gen.plot_burstdata(user_burst_list)
    plt.show()

    # In PyWASPGEN, synthetic radio frequency captures are provided in complex-baseband format where the real-component of the captures represent the in-phase components
    # and the imaginary-components represent the quadrature components (https://en.wikipedia.org/wiki/Baseband#Equivalent_baseband_signal).

    # Here, we create the radio frequency capture from the user-defined signal burst list created above.
    iq_data, updated_burst_list = iq_gen.gen_iqdata([user_burst_list])

    # Print the returned burst list and plot the spectrogram of the created radio frequency capture with and without the burst metadata overlaid.
    print(updated_burst_list)
    ax = burst_gen.plot_burstdata(updated_burst_list[0])
    iq_gen.plot_iqdata(iq_data[0], ax)
    plt.show()
    iq_gen.plot_iqdata(iq_data[0])
    plt.show()