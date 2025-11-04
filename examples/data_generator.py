"""
An example script for showing how to randomly generate burst definition objects using burst_datagen for use in iq_datagen to create in-phase/quadrature (IQ) data.
"""

import argparse
import pickle
import signal
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import pywaspgen.impairments as impairments
from pywaspgen.burst_datagen import BurstDatagen
from pywaspgen.iq_datagen import IQDatagen

# Just for sync / gracefull spindown
is_alive = True

label_dict = {
    "BASK": 0,
    "16ASK": 1,
    "4PAM": 2,
    "16PAM": 3,
    "QPSK": 4,
    "8PSK": 5,
    "16PSK": 6,
    "16QAM": 7,
    "ALT-QPSK": 8,
}


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
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=100,
        help="Generate <terations> amount of data.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        help="Batch the IQ generation into groups of <Batch_Size>",
    )
    return parser.parse_args()


def graph_specgram(iq, fname):
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.set_size_inches(
        1000.0 / plt.rcParams["figure.dpi"], 500.0 / plt.rcParams["figure.dpi"]
    )
    ax.axis("off")
    pxx, freqs, bins, im = ax.specgram(x=iq, Fs=1.0, noverlap=64, NFFT=512)
    ax.axis("off")
    fig.savefig(fname, dpi=640, transparent=True)
    plt.close()


def _gen_save_off(burst_entry, iq_data):
    """
    Illustrates saving the meta data and the RAW IQ for each generate signal.
    We've assumed perfect isolation and shifting down to baseband.
    """

    yolo_label_listing = []
    for b in burst_entry:
        corrected_iq_data = iq_data[
            b.get_start_time() : b.get_end_time()
        ]  # Corrects the sample offset of the received samples.
        corrected_iq_data = impairments.freq_off(
            corrected_iq_data, -b.get_center_freq()
        )  # Corrects the frequency offset of the received samples.

        # Save IQ
        unique_id = str(b.get_meta_uuid())

        np.save(args.folder_path_out + "/iq/" + unique_id, corrected_iq_data)
        with open(args.folder_path_out + "/meta/" + unique_id + ".pkl", "wb") as f:
            pickle.dump(b, f)

        x_cent = b.cent_freq + 0.5
        y_cent = b.get_mid_time() / len(iq_data)
        width = b.bandwidth
        height = b.duration / len(iq_data)
        label = label_dict[b.get_meta_label()]
        annotation = f"{label} {x_cent:.6f} {y_cent:.6f} {width:.6f} {height:.6f}"
        yolo_label_listing.append(annotation)

    # Now Save off the Image
    # Now Save off the Label
    # plt.specgram(iq_data, 256, 1.0, noverlap=64, cmap="plasma")
    graph_specgram(
        iq=iq_data, fname=args.folder_path_out + "/images/" + unique_id + ".png"
    )
    # plt.savefig(args.folder_path_out + "/images/" + unique_id + ".png")
    # plt.close()

    with open(args.folder_path_out + "/labels/" + unique_id + ".txt", "w") as f:
        for annot in yolo_label_listing:
            f.write(annot + "\n")


def main(args):
    # Instantiate a burst generator object and specify the signal parameter configuration file to use.
    burst_gen = BurstDatagen(args.config_file)
    # Instantiate an iq generator object and specify the signal parameter configuration file to use.
    iq_gen = IQDatagen(args.config_file)

    burst_list_complete = []
    # Compile a burst listing
    print(
        "Compiling Burst Listing --- This may take time depending on number of iterations."
    )
    burst_list_complete = burst_gen.gen_burstlist(args.iterations)
    print("Burst Listing Completed.\n")
    print(
        "Beginning IQ Generation with multiprocessing. \n Adjust the generation:pool parameter to increase/decrease threading.\n Again this may take time\n"
    )
    print(
        f"We are Batching this out in size : {args.batch_size} so well have {args.iterations//args.batch_size} loops to generate the full set of data. \n"
    )
    loop_count = args.iterations // args.batch_size
    prev_start = 0

    for i in range(loop_count):
        iq_data, updated_burst_list = iq_gen.gen_iqdata(
            burst_list_complete[prev_start : prev_start + args.batch_size]
        )
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
    file_path_img = args.folder_path_out + "/images/"
    file_path_labels = args.folder_path_out + "/labels/"

    Path(file_path_meta).mkdir(parents=True, exist_ok=True)
    Path(file_path_iq).mkdir(parents=True, exist_ok=True)
    Path(file_path_img).mkdir(parents=True, exist_ok=True)
    Path(file_path_labels).mkdir(parents=True, exist_ok=True)

    main(args)
