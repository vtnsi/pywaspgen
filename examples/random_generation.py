"""
An example script for showing how to ranomdly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""
import matplotlib.pyplot as plt

from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

if __name__ == "__main__":
    # Instantiate a burst generator object and specify the signal parameter configuration file to use.
    burst_gen = BurstDatagen("configs/default.json")
    # Instantiate an iq generator object and specify the signal parameter configuration file to use.

    iq_gen = IQDatagen("configs/default.json")

    # In PyWASPGEN, a burst defines the 'extent' and 'type' of signal to be created.
    #   extent - The time/frequency bounds of the signal to be generated.
    #   type - The signal format and associated parameters of the signal to be generated.
    # Here, we use the instantiated burst generator object to create a random list of signal bursts based on the chosen configuration file (to support repeatability, the configuration file has a random seed parameter that will need to be changed to see a different random list each time this script is run).
    rand_burst_list = burst_gen.gen_burstlist()

    # Print and plot the list of random signal bursts (note that each burst is given a universally unique identifier).
    print("\nUser-Defined Burst List:")
    print(rand_burst_list)
    burst_gen.plot_burstdata(rand_burst_list[0])
    plt.show()

    # In PyWASPGEN, synthetic radio frequency captures are provided in complex-baseband format where the real-component of the captures represent the in-phase components
    # and the imaginary-components represent the quadrature components (https://en.wikipedia.org/wiki/Baseband#Equivalent_baseband_signal).

    # Here, we create the radio frequency capture from the random signal burst list created above.
    iq_data, updated_burst_list = iq_gen.gen_iqdata(rand_burst_list)

    # Print the returned burst list and plot the spectrogram of the created radio frequency capture with and without the burst metadata overlaid (note that the updated burst list now includes the metadata from the iq generator).
    print(updated_burst_list[0])
    ax = burst_gen.plot_burstdata(updated_burst_list[0])
    iq_gen.plot_iqdata(iq_data[0], ax)
    plt.show()
    iq_gen.plot_iqdata(iq_data[0])
    plt.show()