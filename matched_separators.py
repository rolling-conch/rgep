"""Model construction and array fingerprints for matched separator experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.RSSNetInspired import RSSNetInspired


def array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def fresh_paper_separator(template):
    """Copy ONLY the checkpoint graph. model_from_json initializes NEW weights.

    This avoids assuming the local paper model source still matches its checkpoint.
    The parallel classification head is intentionally absent from both competitors.
    """
    import tools
    custom = tools.load_custom_objects("CSAM_ResUNet_classification")
    full = tf.keras.models.load_model(template, custom_objects=custom, compile=False)
    a, b = (full.get_layer(name).output for name in ("spectrum_A", "spectrum_B"))
    endpoints = tf.keras.Model(full.input, tf.keras.layers.Concatenate(name="separated_components")([a, b]))
    def clone_layer(layer):
        config = layer.get_config()
        if layer.name in ("spectrum_A", "spectrum_B"):
            # Match RSSNet's final spectral decoder initialization. Keep the
            # paper graph/activations intact, but avoid scale-biased dead ReLU.
            config["kernel_initializer"] = tf.keras.initializers.serialize(
                tf.keras.initializers.RandomNormal(stddev=1e-3))
            config["bias_initializer"] = tf.keras.initializers.serialize(
                tf.keras.initializers.Constant(0.005))
        return layer.__class__.from_config(config)
    fresh = tf.keras.models.clone_model(endpoints, clone_function=clone_layer)
    fresh.trainable = True
    if fresh.output_shape != (None, 1024, 2):
        raise ValueError("Unexpected paper endpoint shape")
    return fresh


def load_run_model(run, config):
    if config.get("model_kind", "rssnet") == "rssnet":
        from models.RSSNetVariants import build_rssnet
        return build_rssnet(**config["architecture"])
    if config["model_kind"] != "paper":
        raise ValueError("Unknown separator")
    import tools
    from fixed_splits import file_sha256
    if config.get("architecture_sha256") and file_sha256(Path(run) / "architecture.json") != config["architecture_sha256"]:
        raise ValueError("Paper graph changed since training")
    return tf.keras.models.model_from_json(
        (Path(run) / "architecture.json").read_text(encoding="utf-8"),
        custom_objects=tools.load_custom_objects("CSAM_ResUNet_classification"))


def matched_learning_rate(epoch, initial):
    # Fixed epoch schedule, identical for both models, independent of validation.
    return initial * (0.5 ** sum(epoch >= boundary for boundary in (20, 35)))
