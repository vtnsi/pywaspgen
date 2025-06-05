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
    batch_size = 10  # Number of RF spectrum captures to multiprocess.

    burst_gen = BurstDatagen("configs/default.json")
    iq_gen = IQDatagen("configs/default.json")

    start_time = time.time()
    burst_lists = burst_gen.gen_batch(batch_size)
    iq_data, burst_lists = iq_gen.gen_batch(burst_lists)

    print("Generated {} spectrums in {:.3f} seconds.".format(batch_size, time.time() - start_time))
