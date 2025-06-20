"""
A script for testing and timing the batch processing capabilities of the generators.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time

from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

if __name__ == "__main__":
    num_batches = 2
    batch_size = 10

    burst_gen = BurstDatagen("configs/default.json")
    iq_gen = IQDatagen("configs/default.json")

    start_time = time.time()

    for batch_num in range(num_batches):
        burst_lists = burst_gen.gen_burstlist(batch_size)
        iq_data, burst_lists = iq_gen.gen_iqdata(burst_lists)

    print("Generated {} spectrums in {:.3f} seconds.".format(batch_size, time.time() - start_time))