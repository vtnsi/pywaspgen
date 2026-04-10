"""
Configuration file for WiFi/Bluetooth spectrum dataset generation.
Structured to mirror wifi_band.json.
"""

import numpy as np


class DatasetConfig:
    """Configuration for generating WiFi/Bluetooth spectrum dataset."""

    # ==========================================================
    # Generation Parameters
    # ==========================================================

    RAND_SEED = 1234
    POOL_SIZE = 8
    MAX_ATTEMPTS = 100

    TRAIN_SIZE = 10000
    TEST_SIZE = 2000
    VAL_SIZE = 2000
    OOD_SIZE = 2000

    OUTPUT_FILE = "rf_signal_dataset_ism.h5"
    OUTPUT_DIR = "data"

    SEEDS = {
        'train': 42,
        'test': 1337,
        'val': 9999,
        'ood': 7777
    }

    # ==========================================================
    # Spectrum-Level Configuration
    # ==========================================================

    OBSERVATION_DURATION = 10000
    MAX_SIGNALS_MIN = 1
    MAX_SIGNALS_MAX = 100
    ALLOW_COLLISIONS = False

    # ==========================================================
    # Default Burst Parameters
    # ==========================================================

    CENT_FREQ_MIN = -0.3
    CENT_FREQ_MAX = 0.1

    BANDWIDTH_MIN = 0.01
    BANDWIDTH_MAX = 0.4

    BURST_START_MIN = -400
    BURST_START_MAX = 1900

    BURST_DURATION_MIN = 500
    BURST_DURATION_MAX = 2000

    # ==========================================================
    # IQ Parameters
    # ==========================================================

    SNR_MIN = 10.0
    SNR_MAX = 30.0

    # Pulse shaping
    PULSE_TYPE = "rrc"
    PULSE_BETA = 0.35
    PULSE_SPAN = 10
    WINDOW_TYPE = "kaiser"
    WINDOW_PARAM = 5

    # FSK parameter
    FSK_MOD_INDEX = 0.5

    # ==========================================================
    # Signal Definitions (WiFi + Bluetooth)
    # ==========================================================

    SIGNAL_TYPES = [
        {
            "label": "Bluetooth Advertising",
            "modem": "fsk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": [-0.48, -0.24, 0.30],
            "bandwidth": 0.02,
            "start": [-100, 1900],
            "duration_choices": [500, 750, 1000, 1500, 2000]
        },
        {
            "label": "Bluetooth LE",
            "modem": "fsk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": np.round(np.arange(-0.48, 0.31, 0.01), 2).tolist(),
            "bandwidth": 0.02,
            "start": [-100, 1900],
            "duration_choices": [500, 750, 1000, 1500, 2000]
        },
        {
            "label": "WiFi",
            "modem": "psk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": [-0.38, -0.13, 0.12],
            "bandwidth": 0.2,
            "start": [-100, 1900],
            "duration_choices": [500, 1000, 1500, 2000]
        }
    ]

    OOD_SIGNAL_TYPES = [
        {
            "label": "Bluetooth Advertising",
            "modem": "psk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": [-0.48, -0.24, 0.30],
            "bandwidth": 0.02,
            "start": [-100, 1900],
            "duration_choices": [500, 750, 1000, 1500, 2000]
        },
        {
            "label": "Bluetooth LE",
            "modem": "psk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": np.round(np.arange(-0.48, 0.31, 0.01), 2).tolist(),
            "bandwidth": 0.02,
            "start": [-100, 1900],
            "duration_choices": [500, 750, 1000, 1500, 2000]
        },
        {
            "label": "WiFi",
            "modem": "fsk",
            "modem_params": {
                'order': 2,
                'pulse_params': {
                    'type': 'rrc',
                    'beta': 0.35,
                    'span': 10
                }
            },
            "cent_freq": [-0.38, -0.13, 0.12],
            "bandwidth": 0.2,
            "start": [-100, 1900],
            "duration_choices": [500, 1000, 1500, 2000]
        }
    ]

    # ==========================================================
    # Metadata Structure
    # ==========================================================

    @staticmethod
    def get_metadata_dtype():
        """Returns structured dtype for metadata storage."""
        return np.dtype([
            ('sample_id', 'S36'),
            ('split', 'S30'),
            ('snr', np.float32),
            ('center_frequency', np.float32),
            ('bandwidth', np.float32),
            ('burst_start', np.int32),
            ('burst_duration', np.int32),
            ('signal_label', 'S10'),
            ('modulation_label', 'S10'),
            ('modulation_type', 'S10'),
            ('modulation_order', np.int32),
            ('timestamp', 'S26'),
            ('iq_length', np.int32),
        ])

    # ==========================================================
    # Sampling Functions
    # ==========================================================

    @staticmethod
    def sample_snr(rng):
        """Uniform SNR sampling (0–10 dB)."""
        return rng.uniform(
            DatasetConfig.SNR_MIN,
            DatasetConfig.SNR_MAX
        )

    @staticmethod
    def sample_center_frequency(rng, signal_cfg):
        """Sample center frequency from allowed list."""
        return rng.choice(signal_cfg["cent_freq"])

    @staticmethod
    def sample_burst_bandwidth(signal_cfg):
        """Return fixed bandwidth per signal."""
        return signal_cfg["bandwidth"]

    @staticmethod
    def sample_burst_start(rng, signal_cfg):
        """Sample burst start time."""
        return rng.choice(signal_cfg["start"])

    @staticmethod
    def sample_burst_duration(rng, signal_cfg):
        """Sample duration from allowed discrete durations."""
        return rng.choice(signal_cfg["duration_choices"])

    @staticmethod
    def sample_signal_type(rng, ood=False):
        """Randomly choose Bluetooth Advertising, BLE, or WiFi."""
        if ood:
            idx = rng.integers(0, len(DatasetConfig.OOD_SIGNAL_TYPES))
            return DatasetConfig.OOD_SIGNAL_TYPES[idx]
        else:
            idx = rng.integers(0, len(DatasetConfig.SIGNAL_TYPES))
            return DatasetConfig.SIGNAL_TYPES[idx]