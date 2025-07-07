"""
A script for testing and timing the batch processing capabilities of the generators.
"""
import time

from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

if __name__ == "__main__":
    num_batches = 10
    batch_size = 100

    burst_gen = BurstDatagen("configs/default.json")
    iq_gen = IQDatagen("configs/default.json")

    start_time = time.time()

    for batch_num in range(num_batches):
        burst_lists = burst_gen.gen_burstlist(batch_size)
        iq_data, burst_lists = iq_gen.gen_iqdata(burst_lists)

    print("Generated {} batches of {} spectrums in {:.3f} seconds.".format(num_batches, batch_size, time.time() - start_time))