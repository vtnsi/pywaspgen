from jsonschema import validate

SCHEMA = {
    "type": "object",
    "properties": {
        "generation": {
            "type": "object",
            "properties": {
                "rand_seed": {"description": "Seed for the random number generator", "type": "integer"},
                "pool": {"description": "Number of processes to use", "type": "integer", "minimum": 1},
            },
        },
        "spectrum": {
            "type": "object",
            "properties": {
                "observation_duration": {
                    "description": "Observation duration in seconds",
                    "type": "integer",
                    "minimum": 1,
                },
                "sig_types": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string", "enum": ["ask", "pam", "psk", "qam"]},
                            "order": {"type": "integer", "minimum": 2},
                            "label": {"type": "string"},
                        },
                    },
                },
                "max_signals": {
                    "description": "Maximum number of signals to generate",
                    "type": "integer",
                    "minimum": 1,
                },
                "allow_collisions_flag": {"description": "Flag to allow collisions", "type": "boolean"},
                "max_attempts": {
                    "description": "Maximum number of attempts to generate a signal",
                    "type": "integer",
                    "minimum": 1,
                },
            },
        },
        "burst_defaults": {
            "type": "object",
            "properties": {
                "cent_freq": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1}},
                "bandwidth": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}},
                "start": {"type": "array", "items": {"type": "integer"}},
                "duration": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            },
        },
        "iq_defaults": {
            "type": "object",
            "properties": {"snr": {"type": "array", "items": {"type": "number", "minimum": 0}}},
        },
        "pulse_shape_defaults": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["RRC"]},
                "beta": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}},
                "span": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "window": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["kaiser"]},
                        "params": {"type": "number", "minimum": 0},
                    },
                },
            },
        },
    },
}


def dynamic_validation(config):
    """
    Dynamically perform additional validation
    """
    pass


def validate_schema(config):
    validate(config, SCHEMA)
    dynamic_validation(config)
