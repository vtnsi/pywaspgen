"""
An example script for showing how to directly generate burst definition objects for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import matplotlib.pyplot as plt

from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.burst_def import BurstDef
from pywaspgen.iq_datagen import IQDatagen

# Instantiate a burst generator object (note: here only to be used for plotting the burst list).
burst_gen = BurstDatagen()

# Instantiate an iq generator object and specify the signal parameter configuration file to use (note: directly setting iq specific signal parameters will be added in a future update).
iq_gen = IQDatagen("configs/default.json")

# In PyWASPGEN, a burst defines the 'extent' and 'type' of signal to be created.
#   extent - The time/frequency bounds of the signal to be generated.
#   type - The signal format and associated parameters of the signal to be generated.
# Here, we create a list of three user-defined signal bursts (for a complete list of the signal parameters and their ranges, see doc/parameters.txt).
user_burst_list = [
    BurstDef(
        cent_freq=0.3,
        bandwidth=0.2,
        start=1000,
        duration=50000,
        sig_type={"label": "BASK", "format": "ask", "order": 2},
    )
]
user_burst_list.append(
    BurstDef(
        cent_freq=-0.3,
        bandwidth=0.1,
        start=5000,
        duration=8000,
        sig_type={"label": "QPSK", "format": "psk", "order": 2},
    )
)
user_burst_list.append(
    BurstDef(
        cent_freq=0.0,
        bandwidth=0.35,
        start=10000,
        duration=80000,
        sig_type={"label": "64QAM", "format": "qam", "order": 64},
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
iq_data, updated_burst_list = iq_gen.gen_iqdata(user_burst_list)

# Print the returned burst list and plot the spectrogram of the created radio frequency capture with and without the burst metadata overlaid.
print(updated_burst_list)
ax = burst_gen.plot_burstdata(updated_burst_list)
iq_gen.plot_iqdata(iq_data, ax)
plt.show()
iq_gen.plot_iqdata(iq_data)
plt.show()
