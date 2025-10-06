"""
An example script for showing how to directly generate burst definition objects for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""
from pywaspgen.pywaspgen import Pywaspgen

if __name__ == "__main__":
    # Instantiate a Pywaspgen object which will create an internal burst generator object and an iq generator object
    pywaspgen = Pywaspgen(config_file="configs/default.json", output_directory="output/")

    # In PyWASPGEN, a burst defines the 'extent' and 'type' of signal to be created.
    #   extent - The time/frequency bounds of the signal to be generated.
    #   type - The signal format and associated parameters of the signal to be generated.
    # Here, we create a list of three user-defined signal bursts (for a complete list of the signal parameters and their ranges, see doc/parameters.txt).
    pywaspgen.create_and_add_direct_burst_def(
        cent_freq=0.3,
        bandwidth=0.2,
        start=1000,
        duration=50000,
        sig_type={"type": "ask", "order": 2, "label": "BASK"},
    )
    pywaspgen.create_and_add_direct_burst_def(
        cent_freq=-0.3,
        bandwidth=0.1,
        start=5000,
        duration=8000,
        sig_type={"type": "qam", "order": 16, "label": "16QAM"}
    )

    # Print and plot the list of user-defined signal bursts (note that each burst is given a universally unique identifier).
    print("\nUser-Defined Burst List:")
    print(pywaspgen.burst_list)
    pywaspgen.plot_burst_data()

    # In PyWASPGEN, synthetic radio frequency captures are provided in complex-baseband format where the real-component of the captures represent the in-phase components
    # and the imaginary-components represent the quadrature components (https://en.wikipedia.org/wiki/Baseband#Equivalent_baseband_signal).

    # Here, we create the radio frequency capture from the user-defined signal burst list created above.
    pywaspgen.generate_iq_data()

    # Print the returned burst list and plot the spectrogram of the created radio frequency capture with and without the burst metadata overlaid.
    print(pywaspgen.updated_burst_list)
    pywaspgen.plot_burst_data(updated=True)
    pywaspgen.plot_iq_data()
    pywaspgen.plot_iq_data(with_burst=False)
    