"""
An example script for showing how to ranomdly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import argparse
import multiprocessing
import pickle
import signal
from itertools import repeat
from pathlib import Path

import numpy as np

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

# Just for Sync / Gracefull spindown
is_alive = True


def signal_handler(sig, frame):
    global is_alive
    is_alive = False


signal.signal(signal.SIGINT, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder-path-out",
        dest="folder_path_out",
        type=str,
        default="/tmp/dataset_gen",
        help="Path to save location",
    )
    parser.add_argument(
        "--config-file",
        dest="config_file",
        type=str,
        default="configs/default.json",
        help="Config to use",
    )
    parser.add_argument("--iterations", dest="iterations", type=int, default=1000, help="Generate <terations> amount of data.")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=10, help="Number of Workers")
    parser.add_argument("--gpu", dest="gpu", type=int, default=-1, help="GPU to use")

    return parser.parse_args()


def _gen_save_off(burst_entry, iq_data):
    corrected_iq_data = iq_data[burst_entry.get_start_time() : burst_entry.get_end_time()]  # Corrects the sample offset of the received samples.
    corrected_iq_data = impairments.freq_off(corrected_iq_data, -burst_entry.get_center_freq())  # Corrects the frequency offset of the received samples.

    # Save IQ
    unique_id = str(burst_entry.get_meta_uuid())
    # print(f'UUID: {unique_id}')
    # Assume the Following Structure
    # <datapath>/meta
    # <datapath>/iq

    np.save(args.folder_path_out + "/iq/" + unique_id, corrected_iq_data)
    with open(args.folder_path_out + "/meta/" + unique_id + ".pkl", "wb") as f:
        pickle.dump(burst_entry, f)


def main(args):
    # Instantiate a burst generator object and specify the signal parameter configuration file to use.
    burst_gen = BurstDatagen("configs/default.json")
    # Instantiate an iq generator object and specify the signal parameter configuration file to use.
    iq_gen = IQDatagen("configs/default.json")

    counter = 0

    # create a default process pool
    pool = multiprocessing.Pool(args.num_workers)

    while is_alive and counter < args.iterations:
        # Generate Batch of Bursts
        burst_list = burst_gen.gen_burstlist()
        # Here, we create the radio frequency capture from the random signal burst list created above.
        iq_data, updated_burst_list = iq_gen.gen_iqdata(burst_list=burst_list)

        pool.starmap(_gen_save_off, zip(updated_burst_list, repeat(iq_data)))

        if counter % 10 == 0 and counter > 0:
            print(f"------ Iteration {counter} of {args.iterations} ------ \n")

        """

        for burst_entry in updated_burst_list:

            #print(f'Burst Entry: {burst_entry} : {burst_entry.start} : {burst_entry.start + burst_entry.duration}')
            # Here is where we could permute the 'boxed' IQ for a known CF shift or Time Shift
            corrected_iq_data = iq_data[burst_entry.get_start_time() : burst_entry.get_end_time()]  # Corrects the sample offset of the received samples.
            corrected_iq_data = impairments.freq_off(corrected_iq_data, -burst_entry.get_center_freq())  # Corrects the frequency offset of the received samples.

            # Save IQ
            unique_id = str(burst_entry.get_meta_uuid())
            #print(f'UUID: {unique_id}')
            # Assume the Following Structure
            # <datapath>/meta
            # <datapath>/iq

            np.save(args.folder_path_out+'/iq/'+ unique_id,corrected_iq_data)
            with open(args.folder_path_out+'/meta/'+ unique_id + '.pkl', 'wb') as f:
                pickle.dump(burst_entry, f)
        """
        # pool.join()

        counter = counter + 1

    pool.close()
    # block until all tasks are complete and processes close
    pool.join()
    print("Shutting down the data generations now \n")
    print(f"Data is located in {args.folder_path_out} \n")


if __name__ == "__main__":
    args = parse_args()
    file_path_meta = args.folder_path_out + "/meta/"
    file_path_iq = args.folder_path_out + "/iq/"

    Path(file_path_meta).mkdir(parents=True, exist_ok=True)
    Path(file_path_iq).mkdir(parents=True, exist_ok=True)

    main(args)
