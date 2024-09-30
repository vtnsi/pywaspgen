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
    batch_size = 10

    burst_gen = BurstDatagen("configs/default.json")
    iq_gen = IQDatagen("configs/default.json")

    start_time = time.time()
    burst_lists = burst_gen.gen_batch(batch_size)
    iq_data = iq_gen.gen_batch(burst_lists)

    print("Generated {} signals in {:.3f} seconds.".format(batch_size, time.time() - start_time))
