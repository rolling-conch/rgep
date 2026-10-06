"""Independent SpEx-inspired Raman target extraction, not official SpEx+."""
import tensorflow as tf
from models.RSSNetInspired import GlobalNorm, DualPathBlock

@tf.keras.utils.register_keras_serializable(package='Raman')
class ReferenceGuidedExtractor(tf.keras.Model):
    def __init__(self, channels=64, blocks=3, heads=4, depth=2, chunk_size=32, version=1, **kwargs):
        super().__init__(**kwargs)
        if min(channels, blocks, heads, depth, chunk_size) < 1 or channels % heads or chunk_size % 2:
            raise ValueError('Invalid architecture')
        self.architecture = dict(channels=channels, blocks=blocks, heads=heads, depth=depth, chunk_size=chunk_size)
        if version not in (1, 2):
            raise ValueError('Unknown extractor version')
        self.version = version
        if version == 2:
            self.architecture['version'] = version
        self.encoder = tf.keras.layers.Conv1D(channels, 3, padding='same', activation='relu')
        self.norm = GlobalNorm()
        # Preserve coarse spectral-position information in the reference descriptor.
        self.embedding = tf.keras.layers.Dense(channels, activation='relu')
        self.film = tf.keras.layers.Dense(2 * channels)
        self.separator = [DualPathBlock(channels, heads, depth) for _ in range(blocks)]
        self.mask = tf.keras.layers.Conv1D(channels, 1, activation='sigmoid')
        self.decoder = tf.keras.layers.Conv1DTranspose(1, 3, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.001),
            bias_initializer=tf.keras.initializers.Constant(.005 if version == 1 else -1.0))
        self.presence_hidden = tf.keras.layers.Dense(channels, activation='relu')
        self.presence = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x, reference = inputs['mixture'], inputs['reference']
        h = self.norm(self.encoder(x))
        r = self.encoder(reference)
        # Mean-pool contiguous spectral bins; fully differentiable on TF 2.10.
        length = tf.shape(r)[1]
        r = tf.stack([tf.reduce_mean(r[:, i * length // 8:(i + 1) * length // 8], axis=1)
                      for i in range(8)], axis=1)
        embedding = self.embedding(tf.reshape(r, [-1, 8 * self.architecture['channels']]))
        gamma, beta = tf.split(self.film(embedding), 2, axis=-1)
        z = h * (1 + tf.tanh(gamma[:, None])) + beta[:, None]
        k = self.architecture['chunk_size']
        z = tf.signal.frame(z, k, k // 2, pad_end=True, axis=1)
        for block in self.separator:
            z = block(z, training=training)
        z = tf.transpose(z, [0, 3, 1, 2])
        count = tf.signal.overlap_and_add(tf.ones_like(z[:, :1]), k // 2)
        z = tf.transpose(tf.signal.overlap_and_add(z, k // 2) / count, [0, 2, 1])[:, :tf.shape(x)[1]]
        features = [tf.reduce_mean(z, axis=1), tf.reduce_max(z, axis=1), embedding]
        if self.version == 2:
            # Position-preserving mixture/reference evidence, not only pooled LN features.
            # Linear interpolation supports arbitrary spectral lengths on TF 2.10.
            xb = tf.image.resize(x[:, :, None, :], [128, 1])[:, :, 0, 0]
            rb = tf.image.resize(reference[:, :, None, :], [128, 1])[:, :, 0, 0]
            features.extend([xb * 32., rb * 32., xb * rb * 1024., (xb-rb) * 32.])
        p = self.presence(self.presence_hidden(tf.concat(features, axis=-1)))
        logits = self.decoder(h * self.mask(z))
        # V2 has no hard-ReLU dead region and no multiplicative presence shortcut.
        spectrum = (tf.nn.relu(logits) * p[:, None, :] if self.version == 1
                    else .01 * tf.nn.softplus(logits))
        return dict(spectrum=spectrum, presence=p)

    def get_config(self):
        return dict(super().get_config(), **self.architecture)

@tf.keras.utils.register_keras_serializable(package='Raman')
def target_spectral_loss(truth, prediction):
    # Unit-L2 mixture scale; finite absent-target penalty, no SI-SNR on zero truth.
    energy = tf.reduce_sum(tf.square(truth), axis=[1, 2])
    pred_energy = tf.reduce_sum(tf.square(prediction), axis=[1, 2])
    denom = tf.maximum(energy, .01)
    mse = tf.reduce_sum(tf.square(truth - prediction), axis=[1, 2]) / denom
    cos = tf.reduce_sum(truth * prediction, axis=[1, 2]) / tf.sqrt((energy + 1e-8) * (pred_energy + 1e-8))
    diff = (truth[:, 1:] - truth[:, :-1]) - (prediction[:, 1:] - prediction[:, :-1])
    derivative = tf.reduce_sum(tf.square(diff), axis=[1, 2]) / denom
    return mse + .2 * tf.cast(energy > 1e-8, truth.dtype) * (1 - cos) + .1 * derivative


@tf.keras.utils.register_keras_serializable(package='Raman')
def balanced_presence_loss(truth, prediction):
    weights = truth * (7. / 4.) + (1. - truth) * (7. / 10.)
    return tf.reduce_mean(weights, axis=-1) * tf.keras.losses.binary_crossentropy(truth, prediction)


@tf.keras.utils.register_keras_serializable(package='Raman')
def balanced_target_loss(truth, prediction):
    present = tf.reduce_sum(tf.square(truth), axis=[1, 2]) > 1e-8
    # Equalize positive/negative query population, then reduce absent spectral
    # suppression: former 1/.01=100 penalty dominated near-zero positive outputs.
    positive = target_spectral_loss(truth, prediction) * (7. / 4.)
    absent = tf.reduce_sum(tf.square(prediction), axis=[1, 2]) * .5 * (7. / 10.)
    return tf.where(present, positive, absent)
