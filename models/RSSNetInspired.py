"""Independent TensorFlow 2.10 RSSNet-inspired TWO-source separator.

Not official RSSNet: compact widths, normalized overlap-add, nonnegative output,
and amplitude-aware PIT loss are deliberate changes. See RSSNET_INSPIRED.md.
"""
import tensorflow as tf

register = tf.keras.utils.register_keras_serializable


@register(package="Raman")
class GlobalNorm(tf.keras.layers.Layer):
    def build(self, shape):
        self.gain = self.add_weight("gain", shape=(shape[-1],), initializer="ones")
        self.bias = self.add_weight("bias", shape=(shape[-1],), initializer="zeros")

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + 1e-6) * self.gain + self.bias


@register(package="Raman")
class TopDownAttention(tf.keras.layers.Layer):
    """Multi-scale convolution -> pooled MHA -> coarse-to-fine gated fusion."""
    def __init__(self, channels, heads=4, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.heads, self.depth = channels, heads, depth
        self.down = [tf.keras.layers.Conv1D(channels, 3, strides=2, padding="same",
                                           activation="relu") for _ in range(depth)]
        self.attention = tf.keras.layers.MultiHeadAttention(heads, channels // heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(channels * 2, activation="relu"),
                                        tf.keras.layers.Dense(channels)])
        self.gates = [tf.keras.layers.Conv1D(channels, 1, activation="sigmoid")
                      for _ in range(depth + 1)]
        self.biases = [tf.keras.layers.Conv1D(channels, 1) for _ in range(depth + 1)]

    @staticmethod
    def resize(x, length):
        return tf.image.resize(x[:, :, None, :], [length, 1])[:, :, 0, :]

    def call(self, x, training=None):
        scales = [x]
        for layer in self.down:
            scales.append(layer(scales[-1]))
        coarse = tf.add_n([self.resize(s, tf.shape(scales[-1])[1]) for s in scales])
        z = self.norm1(coarse + self.attention(coarse, coarse, training=training))
        z = self.norm2(z + self.ffn(z, training=training))
        for i in reversed(range(len(scales))):
            z = self.resize(z, tf.shape(scales[i])[1])
            z = scales[i] * self.gates[i](z) + self.biases[i](z)
        return z

    def get_config(self):
        return dict(super().get_config(), channels=self.channels, heads=self.heads, depth=self.depth)


@register(package="Raman")
class DualPathBlock(tf.keras.layers.Layer):
    def __init__(self, channels, heads=4, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.heads, self.depth = channels, heads, depth
        self.intra = TopDownAttention(channels, heads, depth)
        self.inter = TopDownAttention(channels, heads, depth)
        self.local = tf.keras.layers.DepthwiseConv2D(3, padding="same")
        self.cross = tf.keras.layers.DepthwiseConv2D(3, padding="same")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, h, training=None):
        # B,T,K,N: T overlapping chunks, each containing K positions.
        b, t, k = tf.shape(h)[0], tf.shape(h)[1], tf.shape(h)[2]
        z = self.intra(tf.reshape(h, [b * t, k, self.channels]), training=training)
        h = self.norm1(h + self.local(h) + tf.reshape(z, [b, t, k, self.channels]))
        z = tf.reshape(tf.transpose(h, [0, 2, 1, 3]), [b * k, t, self.channels])
        z = self.inter(z, training=training)
        z = tf.transpose(tf.reshape(z, [b, k, t, self.channels]), [0, 2, 1, 3])
        return self.norm2(h + self.cross(h) + z)

    def get_config(self):
        return dict(super().get_config(), channels=self.channels, heads=self.heads, depth=self.depth)


@register(package="Raman")
class RSSNetInspired(tf.keras.Model):
    def __init__(self, channels=64, blocks=3, chunk_size=32, heads=4, depth=2, **kwargs):
        super().__init__(**kwargs)
        if min(channels, blocks, chunk_size, heads, depth) < 1 or channels % heads or chunk_size % 2:
            raise ValueError("Positive sizes required; channels divisible by heads, chunk_size even")
        self.channels, self.blocks = channels, blocks
        self.chunk_size, self.heads, self.depth = chunk_size, heads, depth
        self.encoder = tf.keras.layers.Conv1D(channels, 3, padding="same")
        self.gln = GlobalNorm()
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1])
        self.separator = [DualPathBlock(channels, heads, depth) for _ in range(blocks)]
        self.mask_net = tf.keras.layers.Conv1D(2 * channels, 1, activation="sigmoid")
        # Unit-L2 1024-band inputs have small amplitudes. Standard Glorot here
        # produces O(1) spectra after GLN and a huge initial relative-MSE loss.
        self.decoder = tf.keras.layers.Conv1DTranspose(
            1, 3, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(0.005))

    def encode_features(self, x, training=None):
        """Return encoded and merged features without changing separator weights."""
        h = self.prelu(self.gln(self.encoder(x)))
        stride = self.chunk_size // 2
        chunks = tf.signal.frame(h, self.chunk_size, stride, pad_end=True, axis=1)
        z = chunks
        for block in self.separator:
            z = block(z, training=training)
        z = tf.transpose(z, [0, 3, 1, 2])
        merged = tf.signal.overlap_and_add(z, stride)
        count = tf.signal.overlap_and_add(tf.ones_like(z[:, :1]), stride)
        merged = tf.transpose(merged / count, [0, 2, 1])[:, :tf.shape(h)[1]]
        return h, merged

    def decode_features(self, h, merged):
        masks = tf.reshape(self.mask_net(merged), [tf.shape(h)[0], tf.shape(h)[1], 2, self.channels])
        spectra = [self.decoder(h * masks[:, :, i]) for i in range(2)]
        # ReLU with positive bias avoids allowing physically negative components.
        return tf.nn.relu(tf.concat(spectra, axis=-1))

    def call(self, x, training=None):
        h, merged = self.encode_features(x, training=training)
        return self.decode_features(h, merged)

    def get_config(self):
        return dict(super().get_config(), channels=self.channels, blocks=self.blocks,
                    chunk_size=self.chunk_size, heads=self.heads, depth=self.depth)


def assignment_cost(truth, prediction):
    """Per-sample amplitude, shape and derivative error for one ordering."""
    eps = tf.constant(1e-8, truth.dtype)
    energy = tf.reduce_sum(tf.square(truth), axis=1) + eps
    relative_mse = tf.reduce_sum(tf.square(truth - prediction), axis=1) / energy
    cosine = tf.reduce_sum(truth * prediction, axis=1) / tf.sqrt(
        energy * (tf.reduce_sum(tf.square(prediction), axis=1) + eps))
    td = truth[:, 1:] - truth[:, :-1]
    pd = prediction[:, 1:] - prediction[:, :-1]
    derivative = tf.reduce_sum(tf.square(td - pd), axis=1) / energy
    return tf.reduce_mean(relative_mse + 0.2 * (1 - cosine) + 0.1 * derivative, axis=-1)


@register(package="Raman")
def pit_spectral_loss(truth, prediction):
    """Choose source ordering ONCE per sample, not separately at each band."""
    direct = assignment_cost(truth, prediction)
    swapped = assignment_cost(truth, tf.reverse(prediction, axis=[-1]))
    clean = tf.reduce_sum(truth, axis=-1)
    reconstructed = tf.reduce_sum(prediction, axis=-1)
    consistency = tf.reduce_sum(tf.square(clean - reconstructed), axis=1) / (
        tf.reduce_sum(tf.square(clean), axis=1) + 1e-8)
    return tf.minimum(direct, swapped) + 0.1 * consistency
