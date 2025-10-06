"""
An example script for showing how to randomly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen
from pywaspgen.pywaspgen import Pywaspgen
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
    parser.add_argument("--iterations", dest="iterations", type=int, default=100, help="Generate <iterations> amount of data.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="Batch the IQ generation into groups of <Batch_Size>")
    return parser.parse_args()

def main(args):
    # Instantiate a burst generator object and specify the signal parameter configuration file to use.
    pywaspgen = Pywaspgen(config_file=args.config_file, output_directory=args.folder_path_out, iterations=args.iterations)

    # # Compile a burst listing
    print("Compiling Burst Listing --- This may take time depending on number of iterations.")
    pywaspgen.create_random_burst_def()
    print("Burst Listing Completed.\n")

    # Generate and save IQ
    print("Beginning IQ Generation with multiprocessing. \n Adjust the generation:pool parameter to increase/decrease threading.\n Again this may take time\n")
    print(f"We are Batching this out in size : {args.batch_size} so well have {args.iterations // args.batch_size} loops to generate the full set of data. \n")
    pywaspgen.generate_iq_data(batch_size=args.batch_size, random=True, saving=True)

if __name__ == "__main__":
    args = parse_args()
    file_path_meta = args.folder_path_out + "/meta/"
    file_path_iq = args.folder_path_out + "/iq/"

    # Create output directories
    Path(file_path_meta).mkdir(parents=True, exist_ok=True)
    Path(file_path_iq).mkdir(parents=True, exist_ok=True)

    main(args)
