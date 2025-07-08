"""
An example script for showing how to randomly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen
import argparse
import pickle
from pathlib import Path
import numpy as np

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
    parser.add_argument("--iterations", dest="iterations", type=int, default=100, help="Generate <terations> amount of data.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="Batch the IQ generation into groups of <Batch_Size>")
    return parser.parse_args()

def _gen_save_off(burst_entry, iq_data):
    """
    Illustrates saving the meta data and the RAW IQ for each generate signal.
    We've assumed perfect isolation and shifting down to baseband.
    """

    for b in burst_entry:
        corrected_iq_data = iq_data[b.get_start_time() : b.get_end_time()]  # Corrects the sample offset of the received samples.
        corrected_iq_data = impairments.freq_off(corrected_iq_data, -b.get_center_freq())  # Corrects the frequency offset of the received samples.

        # Save IQ
        unique_id = str(b.get_meta_uuid())

        np.save(args.folder_path_out + "/iq/" + unique_id, corrected_iq_data)
        with open(args.folder_path_out + "/meta/" + unique_id + ".pkl", "wb") as f:
            pickle.dump(b, f)

def main(args):
    # Instantiate a burst generator object and specify the signal parameter configuration file to use.
    burst_gen = BurstDatagen(args.config_file)
    # Instantiate an iq generator object and specify the signal parameter configuration file to use.
    iq_gen = IQDatagen(args.config_file)

    burst_list_complete = []
    # Compile a burst listing
    print("Compiling Burst Listing --- This may take time depending on number of iterations.")
    burst_list_complete = burst_gen.gen_burstlist(args.iterations)
    print("Burst Listing Completed.\n")
    print("Beginning IQ Generation with multiprocessing. \n Adjust the generation:pool parameter to increase/decrease threading.\n Again this may take time\n")
    print(f"We are Batching this out in size : {args.batch_size} so well have {args.iterations//args.batch_size} loops to generate the full set of data. \n")
    loop_count = args.iterations // args.batch_size
    prev_start = 0

    for i in range(loop_count):
        iq_data, updated_burst_list = iq_gen.gen_iqdata(burst_list_complete[prev_start : prev_start + args.batch_size])
        print(f"Saving off this batch of IQ : {i} of {loop_count} \n")
        for k in range(len(updated_burst_list)):
            _gen_save_off(burst_entry=updated_burst_list[k], iq_data=iq_data[k])
        prev_start = prev_start + args.batch_size

if __name__ == "__main__":
    # Create the Following Structure
    # <datapath>/meta
    # <datapath>/iq

    args = parse_args()
    file_path_meta = args.folder_path_out + "/meta/"
    file_path_iq = args.folder_path_out + "/iq/"

    Path(file_path_meta).mkdir(parents=True, exist_ok=True)
    Path(file_path_iq).mkdir(parents=True, exist_ok=True)

    main(args)
