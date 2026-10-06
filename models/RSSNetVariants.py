"""Experimental R1/R2/R3 ablations; R0 remains the unchanged RSSNetInspired."""
import tensorflow as tf
from models.RSSNetInspired import RSSNetInspired, DualPathBlock, TopDownAttention

register = tf.keras.utils.register_keras_serializable


@register(package="Raman")
class LightweightIntra(tf.keras.layers.Layer):
    """Noncausal gated separable convolutions within each spectral chunk."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.filters = [tf.keras.layers.SeparableConv1D(
            2 * channels, 3, padding='same', dilation_rate=d) for d in (1, 2, 4)]
        self.output_projection = tf.keras.layers.Conv1D(channels, 1)

    def call(self, x, training=None):
        z = self.norm(x)
        for layer in self.filters:
            value, gate = tf.split(layer(z), 2, axis=-1)
            z = z + tf.tanh(value) * tf.sigmoid(gate)
        return self.output_projection(z)

    def get_config(self):
        return dict(super().get_config(), channels=self.channels)


@register(package="Raman")
class TDAWithFSMN(tf.keras.layers.Layer):
    """Original intra TDA plus a compact, bidirectional FIR memory branch.

    The projected memory uses four preceding/following bins within each chunk.
    Zero output projection initially leaves the TDA output unchanged. This is
    an FSMN-inspired finite-memory branch, not an official FSMN implementation.
    """
    def __init__(self, channels, heads=4, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.heads, self.depth = channels, heads, depth
        self.tda = TopDownAttention(channels, heads, depth)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.projection = tf.keras.layers.Conv1D(max(1, channels // 4), 1, activation='relu')
        self.memory = tf.keras.layers.DepthwiseConv2D((9, 1), padding='same', use_bias=False)
        self.output_projection = tf.keras.layers.Conv1D(
            channels, 1, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, x, training=None):
        p = self.projection(self.norm(x))
        memory = self.memory(p[:, :, None, :])[:, :, 0, :]
        return self.tda(x, training=training) + self.output_projection(p + memory)

    def get_config(self):
        return dict(super().get_config(), channels=self.channels, heads=self.heads, depth=self.depth)


@register(package="Raman")
class ContextDualPathBlock(DualPathBlock):
    def __init__(self, channels, heads=4, depth=2, variant='R4', **kwargs):
        if variant not in ('R4', 'R5'):
            raise ValueError('Context block requires R4 or R5')
        super().__init__(channels, heads, depth, **kwargs)
        self.variant = variant
        self.intra = (LightweightIntra(channels) if variant == 'R4'
                      else TDAWithFSMN(channels, heads, depth))

    def get_config(self):
        return dict(super().get_config(), variant=self.variant)


@register(package="Raman")
class MultiScalePeakEncoder(tf.keras.layers.Layer):
    """Parallel narrow/wide/dilated features, fused to the original width."""
    def __init__(self, channels=64, **kwargs):
        super().__init__(**kwargs)
        if channels < 3:
            raise ValueError('Multi-scale encoder requires channels >= 3')
        self.channels = channels
        widths = [channels // 3, channels // 3, channels - 2 * (channels // 3)]
        self.branches = [tf.keras.layers.Conv1D(w, k, dilation_rate=d, padding='same', activation='relu')
                         for w, k, d in zip(widths, (3, 7, 3), (1, 1, 3))]
        self.fusion = tf.keras.layers.Conv1D(channels, 1, padding='same')

    def call(self, x):
        return self.fusion(tf.concat([branch(x) for branch in self.branches], axis=-1))

    def get_config(self):
        return dict(super().get_config(), channels=self.channels)


@register(package="Raman")
class SharedResidualRefiner(tf.keras.layers.Layer):
    """Shared endpoint correction: swap inputs => swap outputs; not hard consistency.

    The last convolution starts at zero, so nonnegative coarse outputs are
    unchanged at initialization. Corrections are bounded by a fraction of the
    input RMS; their sign is unrestricted, final spectra remain nonnegative.
    """
    def __init__(self, channels=16, scale=0.1, **kwargs):
        super().__init__(**kwargs)
        if channels < 1 or scale <= 0:
            raise ValueError('Positive refinement channels and scale required')
        self.channels, self.scale = channels, scale
        self.local = tf.keras.layers.Conv1D(channels, 3, padding='same', activation='relu')
        self.context = tf.keras.layers.Conv1D(channels, 3, dilation_rate=2, padding='same', activation='relu')
        self.correction = tf.keras.layers.Conv1D(1, 1, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs):
        x, coarse = inputs
        residual = x - tf.reduce_sum(coarse, axis=-1, keepdims=True)
        bound = self.scale * tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-12)
        outputs = []
        for i in range(2):
            own, other = coarse[..., i:i+1], coarse[..., 1-i:2-i]
            features = tf.concat([x, own, other, residual], axis=-1)
            delta = bound * tf.tanh(self.correction(self.context(self.local(features))))
            outputs.append(tf.nn.relu(own + delta))
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        return dict(super().get_config(), channels=self.channels, scale=self.scale)


@register(package="Raman")
class RSSNetVariant(RSSNetInspired):
    def __init__(self, variant='R1', refinement_channels=16, refinement_scale=0.1, **kwargs):
        if variant not in ('R1', 'R2', 'R3', 'R4', 'R5'):
            raise ValueError('RSSNetVariant requires R1-R5; use factory for R0')
        super().__init__(**kwargs)
        self.variant = variant
        self.refinement_channels, self.refinement_scale = refinement_channels, refinement_scale
        if variant in ('R1', 'R3'):
            self.encoder = MultiScalePeakEncoder(self.channels, name='multiscale_peak_encoder')
        self.refiner = SharedResidualRefiner(refinement_channels, refinement_scale, name='shared_refiner') if variant in ('R2', 'R3') else None
        if variant in ('R4', 'R5'):
            self.separator = [ContextDualPathBlock(self.channels, self.heads, self.depth, variant)
                              for _ in range(self.blocks)]

    def call(self, x, training=None):
        coarse = super().call(x, training=training)
        return self.refiner([x, coarse]) if self.refiner is not None else coarse

    def get_config(self):
        return dict(super().get_config(), variant=self.variant, refinement_channels=self.refinement_channels,
                    refinement_scale=self.refinement_scale)


def build_rssnet(variant='R0', refinement_channels=16, refinement_scale=0.1, **architecture):
    if variant == 'R0':
        return RSSNetInspired(**architecture)
    return RSSNetVariant(variant=variant, refinement_channels=refinement_channels,
                         refinement_scale=refinement_scale, **architecture)
