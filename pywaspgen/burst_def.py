"""
This module defines the structure and behavior of bursts via the :class:`BurstDef` object.
"""

import uuid

class BurstDef:
    """
    Used to create burst definition objects that specify the time and frequency extent of a burst as well as the burst's metadata. These objects are passed as input to :class:`pywaspgen.iq_datagen.IQDatagen` to create in-phase/quadrature (IQ) data.
    """

    def __init__(self, cent_freq, bandwidth, start, duration, sig_type, metadata={}):
        """
        The constructor for the burst_def class.

        Args:
            cent_freq (float): The normalized center frequency, or mid-point in frequency, of the burst (in Hz).
            bandwidth (float): The normalized bandwidth of the burst (in Hz).
            start (int): The normalized start time of the burst (in sec).
            duration (int): The duration of the burst (in sec).
            sig_type (dict): The signal type of the burst.
            metadata (dict): The signal metadata of the burst.
        """
        self.cent_freq = cent_freq
        self.bandwidth = bandwidth
        self.start = start
        self.duration = duration
        self.sig_type = sig_type
        self.metadata = metadata.copy()

        # Assigns a unique identifier to the burst if not provided by the user.
        if "UUID" not in self.metadata.keys():
            self.metadata["UUID"] = uuid.uuid4()

    def get_low_freq(self):
        """
        Returns:
            float: The lower edge of the bandwidth of the burst (in Hz).
        """
        return self.cent_freq - self.bandwidth / 2.0

    def get_high_freq(self):
        """
        Returns:
            float: The higher edge of the bandwidth of the burst (in Hz).
        """
        return self.cent_freq + self.bandwidth / 2.0

    def get_center_freq(self):
        return self.cent_freq

    def get_mid_time(self):
        """
        Returns:
            int: The center time, or mid-time, of the burst (in seconds).
        """
        return self.start + self.duration / 2.0

    def get_end_time(self):
        """
        Returns:
            int: The end time of the burst (in seconds).
        """
        return self.start + self.duration

    def get_start_time(self):
        """
        Returns:
           int: The start time of the burst (in seconds).
        """
        return self.start

    def get_meta_uuid(self):
        return self.metadata["UUID"]

    def get_meta_data(self):
        return self.metadata

    def get_meta_label(self) -> str:
        return self.sig_type["label"]

    def __repr__(self):
        return f"\nUUID: {self.metadata['UUID']} - {self.sig_type['label']} burst with center frequency: {self.cent_freq:.3f}, bandwidth: {self.bandwidth:.3f}, start time: {self.start}, duration: {self.duration}, and metadata: {self.metadata}"
