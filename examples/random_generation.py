
"""
An example script for showing how to ranomdly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

from pywaspgen.pywaspgen import Pywaspgen

if __name__ == "__main__":
    pywaspgen = Pywaspgen(config_file="configs/default.json", output_directory="output/")

    # In PyWASPGEN, a burst defines the 'extent' and 'type' of signal to be created.
    #   extent - The time/frequency bounds of the signal to be generated.
    #   type - The signal format and associated parameters of the signal to be generated.
    # Here, we use the instantiated burst generator object to create a random list of signal bursts based on the chosen configuration file (to support repeatability, the configuration file has a random seed parameter that will need to be changed to see a different random list each time this script is run).
    pywaspgen.create_random_burst_def()

    # Print and plot the list of random signal bursts (note that each burst is given a universally unique identifier).
    print("\nUser-Defined Burst List:")
    print(pywaspgen.burst_list)
    pywaspgen.plot_burst_data(random=True)

    # In PyWASPGEN, synthetic radio frequency captures are provided in complex-baseband format where the real-component of the captures represent the in-phase components
    # and the imaginary-components represent the quadrature components (https://en.wikipedia.org/wiki/Baseband#Equivalent_baseband_signal).

    # Here, we create the radio frequency capture from the random signal burst list created above.
    pywaspgen.generate_iq_data(random=True)

    # Print the returned burst list and plot the spectrogram of the created radio frequency capture with and without the burst metadata overlaid (note that the updated burst list now includes the metadata from the iq generator).
    print(pywaspgen.updated_burst_list)
    pywaspgen.plot_burst_data(updated=True)
    pywaspgen.plot_iq_data()
    pywaspgen.plot_iq_data(with_burst=False)