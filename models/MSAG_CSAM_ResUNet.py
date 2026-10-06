"""Multi-scale attention-gated CSAM-ResUNet models for Raman spectra.

The model keeps the encoder/decoder layout of the original CSAM-ResUNet while
adding spectral multi-scale convolutions, corrected channel attention, gated
skip connections, and a peak-preserving derivative loss.  The unmixing model
adds permutation-invariant component supervision and a recognition head that
reads the predicted component spectra instead of bypassing them at the
bottleneck.

The ``unmixing_only`` variant removes the recognition head entirely for
experiments where classification and spectral separation must be optimized as
independent tasks.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import CosineSimilarity

import tools


@tf.keras.utils.register_keras_serializable(package="CSAMResUNet")
class PeakAwareHybridLoss(tf.keras.losses.Loss):
    """Intensity-weighted reconstruction loss that preserves Raman peaks.

    The intensity weights are calculated inside the loss.  This avoids relying
    on Keras ``sample_weight`` forwarding and ensures that every spectral point
    receives its intended weight.
    """

    def __init__(
        self,
        mse_weight=1.0,
        intensity_weight_scale=100.0,
        intensity_weight_exponent=4.0,
        normalize_intensity=False,
        sparse_strength=0.005,
        derivative_strength=0.05,
        sparse_threshold=1e-4,
        name="peak_aware_hybrid_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mse_weight = float(mse_weight)
        self.intensity_weight_scale = float(intensity_weight_scale)
        self.intensity_weight_exponent = float(intensity_weight_exponent)
        self.normalize_intensity = bool(normalize_intensity)
        self.sparse_strength = float(sparse_strength)
        self.derivative_strength = float(derivative_strength)
        self.sparse_threshold = float(sparse_threshold)

    @staticmethod
    def _first_derivative(tensor):
        return tensor[:, 1:, :] - tensor[:, :-1, :]

    @classmethod
    def _second_derivative(cls, tensor):
        first = cls._first_derivative(tensor)
        return first[:, 1:, :] - first[:, :-1, :]

    def per_sample(self, y_true, y_pred):
        """Return one loss value per spectrum without reducing the batch."""
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon = tf.cast(tf.keras.backend.epsilon(), y_pred.dtype)

        intensity = tf.abs(y_true)
        if self.normalize_intensity:
            # Per-spectrum normalization gives weak components meaningful peak
            # weights even when their absolute amplitude is small.
            scale = tf.reduce_max(intensity, axis=[1, 2], keepdims=True)
            intensity = intensity / (scale + epsilon)
        intensity_weights = 1.0 + self.intensity_weight_scale * tf.pow(
            intensity, self.intensity_weight_exponent
        )
        squared_error = tf.square(y_true - y_pred)
        weighted_mse = tf.reduce_sum(
            intensity_weights * squared_error, axis=[1, 2]
        ) / (tf.reduce_sum(intensity_weights, axis=[1, 2]) + epsilon)

        # Suppress false positive peaks only where the target is effectively 0.
        sparse_mask = tf.cast(
            tf.less_equal(tf.abs(y_true), self.sparse_threshold), y_pred.dtype
        )
        sparse_loss = tf.reduce_sum(
            sparse_mask * tf.square(y_pred), axis=[1, 2]
        ) / (tf.reduce_sum(sparse_mask, axis=[1, 2]) + epsilon)

        # Match target derivatives rather than forcing the prediction itself to
        # be smooth.  This preserves genuine narrow Raman peaks.
        true_d1 = self._first_derivative(y_true)
        pred_d1 = self._first_derivative(y_pred)
        true_d2 = self._second_derivative(y_true)
        pred_d2 = self._second_derivative(y_pred)
        first_derivative_loss = tf.reduce_mean(
            tf.square(true_d1 - pred_d1), axis=[1, 2]
        )
        second_derivative_loss = tf.reduce_mean(
            tf.square(true_d2 - pred_d2), axis=[1, 2]
        )
        derivative_loss = 0.5 * (
            first_derivative_loss + second_derivative_loss
        )

        return (
            self.mse_weight * weighted_mse
            + self.sparse_strength * sparse_loss
            + self.derivative_strength * derivative_loss
        )

    def call(self, y_true, y_pred):
        return self.per_sample(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mse_weight": self.mse_weight,
                "intensity_weight_scale": self.intensity_weight_scale,
                "intensity_weight_exponent": self.intensity_weight_exponent,
                "normalize_intensity": self.normalize_intensity,
                "sparse_strength": self.sparse_strength,
                "derivative_strength": self.derivative_strength,
                "sparse_threshold": self.sparse_threshold,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CSAMResUNet")
class AbsoluteDifference(layers.Layer):
    """Order-invariant absolute difference of two component spectra."""

    def call(self, inputs):
        first, second = inputs
        return tf.abs(first - second)


@tf.keras.utils.register_keras_serializable(package="CSAMResUNet")
class RecognitionGuidedUnmixingModel(tf.keras.Model):
    """Functional model with permutation-invariant unmixing optimization.

    The public inference interface intentionally remains the same as the
    original project: ``[spectrum_A, spectrum_B, classification]``.  PIT,
    mixture consistency, and metric-side component matching are handled only
    inside ``train_step``/``test_step`` so existing prediction scripts do not
    need a fourth output.
    """

    def __init__(
        self,
        *args,
        peak_loss_config=None,
        unmixing_loss_weight=1.0,
        classification_loss_weight=0.01,
        mixture_consistency_weight=0.05,
        classification_label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.peak_loss_config = dict(peak_loss_config or {})
        self.unmixing_loss_weight = float(unmixing_loss_weight)
        self.classification_loss_weight = float(
            classification_loss_weight
        )
        self.mixture_consistency_weight = float(
            mixture_consistency_weight
        )
        self.classification_label_smoothing = float(
            classification_label_smoothing
        )

        self.peak_loss_fn = PeakAwareHybridLoss(**self.peak_loss_config)
        self.classification_loss_fn = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            label_smoothing=self.classification_label_smoothing,
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.unmixing_loss_tracker = tf.keras.metrics.Mean(
            name="unmixing_loss"
        )
        self.mixture_loss_tracker = tf.keras.metrics.Mean(
            name="mixture_consistency_loss"
        )
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.spectrum_a_loss_tracker = tf.keras.metrics.Mean(
            name="spectrum_A_loss"
        )
        self.spectrum_b_loss_tracker = tf.keras.metrics.Mean(
            name="spectrum_B_loss"
        )
        self.swap_rate_tracker = tf.keras.metrics.Mean(
            name="permutation_swap_rate"
        )
        self.spectrum_a_mse = tf.keras.metrics.MeanSquaredError(
            name="spectrum_A_mse"
        )
        self.spectrum_b_mse = tf.keras.metrics.MeanSquaredError(
            name="spectrum_B_mse"
        )
        self.spectrum_a_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectrum_A_mae"
        )
        self.spectrum_b_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectrum_B_mae"
        )
        self.classification_accuracy = tf.keras.metrics.BinaryAccuracy(
            name="classification_binary_accuracy"
        )
        self.classification_precision = tf.keras.metrics.Precision(
            name="classification_precision"
        )
        self.classification_recall = tf.keras.metrics.Recall(
            name="classification_recall"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.unmixing_loss_tracker,
            self.mixture_loss_tracker,
            self.classification_loss_tracker,
            self.spectrum_a_loss_tracker,
            self.spectrum_b_loss_tracker,
            self.swap_rate_tracker,
            self.spectrum_a_mse,
            self.spectrum_b_mse,
            self.spectrum_a_mae,
            self.spectrum_b_mae,
            self.classification_accuracy,
            self.classification_precision,
            self.classification_recall,
        ]

    @staticmethod
    def _unpack_targets(targets):
        if isinstance(targets, dict):
            return (
                targets["spectrum_A"],
                targets["spectrum_B"],
                targets["classification"],
            )
        if isinstance(targets, (tuple, list)) and len(targets) == 3:
            return targets[0], targets[1], targets[2]
        raise ValueError(
            "Unmixing targets must contain spectrum_A, spectrum_B, and "
            "classification."
        )

    def _losses_and_matches(self, targets, predictions):
        true_a, true_b, true_classification = self._unpack_targets(targets)
        pred_a, pred_b, pred_classification = predictions

        loss_aa = self.peak_loss_fn.per_sample(true_a, pred_a)
        loss_bb = self.peak_loss_fn.per_sample(true_b, pred_b)
        loss_ab = self.peak_loss_fn.per_sample(true_a, pred_b)
        loss_ba = self.peak_loss_fn.per_sample(true_b, pred_a)

        direct_loss = loss_aa + loss_bb
        swapped_loss = loss_ab + loss_ba
        use_swapped = swapped_loss < direct_loss
        unmixing_per_sample = tf.minimum(direct_loss, swapped_loss)

        matched_a = tf.where(
            use_swapped[:, tf.newaxis, tf.newaxis], pred_b, pred_a
        )
        matched_b = tf.where(
            use_swapped[:, tf.newaxis, tf.newaxis], pred_a, pred_b
        )
        component_a_loss = tf.reduce_mean(
            tf.where(use_swapped, loss_ab, loss_aa)
        )
        component_b_loss = tf.reduce_mean(
            tf.where(use_swapped, loss_ba, loss_bb)
        )
        unmixing_loss = tf.reduce_mean(unmixing_per_sample)

        # The supplied mixed-spectra training set is exactly X1 + X2.  Keeping
        # this target inside the loss avoids changing the model's three-output
        # inference API.
        true_mixture = true_a + true_b
        pred_mixture = pred_a + pred_b
        mixture_loss = tf.reduce_mean(
            tf.square(true_mixture - pred_mixture)
        )
        classification_loss = tf.reduce_mean(
            self.classification_loss_fn(
                true_classification, pred_classification
            )
        )

        regularization_loss = tf.cast(0.0, unmixing_loss.dtype)
        if self.losses:
            regularization_loss = tf.add_n(self.losses)
        total_loss = (
            self.unmixing_loss_weight * unmixing_loss
            + self.classification_loss_weight * classification_loss
            + self.mixture_consistency_weight * mixture_loss
            + regularization_loss
        )
        return {
            "total": total_loss,
            "unmixing": unmixing_loss,
            "mixture": mixture_loss,
            "classification": classification_loss,
            "component_a": component_a_loss,
            "component_b": component_b_loss,
            "use_swapped": use_swapped,
            "matched_a": matched_a,
            "matched_b": matched_b,
            "true_a": true_a,
            "true_b": true_b,
            "true_classification": true_classification,
            "pred_classification": pred_classification,
        }

    def _update_trackers(self, values):
        self.total_loss_tracker.update_state(values["total"])
        self.unmixing_loss_tracker.update_state(values["unmixing"])
        self.mixture_loss_tracker.update_state(values["mixture"])
        self.classification_loss_tracker.update_state(
            values["classification"]
        )
        self.spectrum_a_loss_tracker.update_state(values["component_a"])
        self.spectrum_b_loss_tracker.update_state(values["component_b"])
        # Pass the per-sample mask so the epoch statistic remains correctly
        # weighted when the final batch is smaller than the others.
        self.swap_rate_tracker.update_state(
            tf.cast(values["use_swapped"], tf.float32)
        )
        self.spectrum_a_mse.update_state(
            values["true_a"], values["matched_a"]
        )
        self.spectrum_b_mse.update_state(
            values["true_b"], values["matched_b"]
        )
        self.spectrum_a_mae.update_state(
            values["true_a"], values["matched_a"]
        )
        self.spectrum_b_mae.update_state(
            values["true_b"], values["matched_b"]
        )
        self.classification_accuracy.update_state(
            values["true_classification"], values["pred_classification"]
        )
        self.classification_precision.update_state(
            values["true_classification"], values["pred_classification"]
        )
        self.classification_recall.update_state(
            values["true_classification"], values["pred_classification"]
        )

    def train_step(self, data):
        inputs, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            values = self._losses_and_matches(targets, predictions)
        gradients = tape.gradient(values["total"], self.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.trainable_variables
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        self._update_trackers(values)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        inputs, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(inputs, training=False)
        values = self._losses_and_matches(targets, predictions)
        self._update_trackers(values)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "peak_loss_config": self.peak_loss_config,
                "unmixing_loss_weight": self.unmixing_loss_weight,
                "classification_loss_weight": (
                    self.classification_loss_weight
                ),
                "mixture_consistency_weight": (
                    self.mixture_consistency_weight
                ),
                "classification_label_smoothing": (
                    self.classification_label_smoothing
                ),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CSAMResUNet")
class PITUnmixingModel(tf.keras.Model):
    """Two-output unmixing model with no classification dependency.

    The component pair is optimized with permutation-invariant training (PIT).
    Classification labels and classification gradients are deliberately absent,
    so the spectral outputs are controlled only by reconstruction, peak-shape,
    sparsity, regularization, and mixture-consistency objectives.
    """

    def __init__(
        self,
        *args,
        peak_loss_config=None,
        unmixing_loss_weight=1.0,
        mixture_consistency_weight=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.peak_loss_config = dict(peak_loss_config or {})
        self.unmixing_loss_weight = float(unmixing_loss_weight)
        self.mixture_consistency_weight = float(
            mixture_consistency_weight
        )
        self.peak_loss_fn = PeakAwareHybridLoss(**self.peak_loss_config)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.unmixing_loss_tracker = tf.keras.metrics.Mean(
            name="unmixing_loss"
        )
        self.mixture_loss_tracker = tf.keras.metrics.Mean(
            name="mixture_consistency_loss"
        )
        self.spectrum_a_loss_tracker = tf.keras.metrics.Mean(
            name="spectrum_A_loss"
        )
        self.spectrum_b_loss_tracker = tf.keras.metrics.Mean(
            name="spectrum_B_loss"
        )
        self.swap_rate_tracker = tf.keras.metrics.Mean(
            name="permutation_swap_rate"
        )
        self.spectrum_a_mse = tf.keras.metrics.MeanSquaredError(
            name="spectrum_A_mse"
        )
        self.spectrum_b_mse = tf.keras.metrics.MeanSquaredError(
            name="spectrum_B_mse"
        )
        self.spectrum_a_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectrum_A_mae"
        )
        self.spectrum_b_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectrum_B_mae"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.unmixing_loss_tracker,
            self.mixture_loss_tracker,
            self.spectrum_a_loss_tracker,
            self.spectrum_b_loss_tracker,
            self.swap_rate_tracker,
            self.spectrum_a_mse,
            self.spectrum_b_mse,
            self.spectrum_a_mae,
            self.spectrum_b_mae,
        ]

    @staticmethod
    def _unpack_targets(targets):
        if isinstance(targets, dict):
            return targets["spectrum_A"], targets["spectrum_B"]
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            return targets[0], targets[1]
        raise ValueError(
            "Decoupled unmixing targets must contain spectrum_A and "
            "spectrum_B only."
        )

    def _losses_and_matches(self, targets, predictions):
        true_a, true_b = self._unpack_targets(targets)
        pred_a, pred_b = predictions

        loss_aa = self.peak_loss_fn.per_sample(true_a, pred_a)
        loss_bb = self.peak_loss_fn.per_sample(true_b, pred_b)
        loss_ab = self.peak_loss_fn.per_sample(true_a, pred_b)
        loss_ba = self.peak_loss_fn.per_sample(true_b, pred_a)

        direct_loss = loss_aa + loss_bb
        swapped_loss = loss_ab + loss_ba
        use_swapped = swapped_loss < direct_loss
        unmixing_loss = tf.reduce_mean(
            tf.minimum(direct_loss, swapped_loss)
        )
        component_a_loss = tf.reduce_mean(
            tf.where(use_swapped, loss_ab, loss_aa)
        )
        component_b_loss = tf.reduce_mean(
            tf.where(use_swapped, loss_ba, loss_bb)
        )
        matched_a = tf.where(
            use_swapped[:, tf.newaxis, tf.newaxis], pred_b, pred_a
        )
        matched_b = tf.where(
            use_swapped[:, tf.newaxis, tf.newaxis], pred_a, pred_b
        )

        true_mixture = true_a + true_b
        pred_mixture = pred_a + pred_b
        mixture_loss = tf.reduce_mean(
            tf.square(true_mixture - pred_mixture)
        )
        regularization_loss = tf.cast(0.0, unmixing_loss.dtype)
        if self.losses:
            regularization_loss = tf.add_n(self.losses)
        total_loss = (
            self.unmixing_loss_weight * unmixing_loss
            + self.mixture_consistency_weight * mixture_loss
            + regularization_loss
        )
        return {
            "total": total_loss,
            "unmixing": unmixing_loss,
            "mixture": mixture_loss,
            "component_a": component_a_loss,
            "component_b": component_b_loss,
            "use_swapped": use_swapped,
            "matched_a": matched_a,
            "matched_b": matched_b,
            "true_a": true_a,
            "true_b": true_b,
        }

    def _update_trackers(self, values):
        self.total_loss_tracker.update_state(values["total"])
        self.unmixing_loss_tracker.update_state(values["unmixing"])
        self.mixture_loss_tracker.update_state(values["mixture"])
        self.spectrum_a_loss_tracker.update_state(values["component_a"])
        self.spectrum_b_loss_tracker.update_state(values["component_b"])
        self.swap_rate_tracker.update_state(
            tf.cast(values["use_swapped"], tf.float32)
        )
        self.spectrum_a_mse.update_state(
            values["true_a"], values["matched_a"]
        )
        self.spectrum_b_mse.update_state(
            values["true_b"], values["matched_b"]
        )
        self.spectrum_a_mae.update_state(
            values["true_a"], values["matched_a"]
        )
        self.spectrum_b_mae.update_state(
            values["true_b"], values["matched_b"]
        )

    def train_step(self, data):
        inputs, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            values = self._losses_and_matches(targets, predictions)
        gradients = tape.gradient(values["total"], self.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.trainable_variables
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        self._update_trackers(values)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        inputs, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(inputs, training=False)
        values = self._losses_and_matches(targets, predictions)
        self._update_trackers(values)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "peak_loss_config": self.peak_loss_config,
                "unmixing_loss_weight": self.unmixing_loss_weight,
                "mixture_consistency_weight": (
                    self.mixture_consistency_weight
                ),
            }
        )
        return config


def channel_attention(inputs, ratio=8):
    """Return a separate attention coefficient for every feature channel."""
    channels = int(inputs.shape[-1])
    hidden_channels = max(channels // ratio, 8)

    average_pool = layers.GlobalAveragePooling1D(keepdims=True)(inputs)
    maximum_pool = layers.GlobalMaxPooling1D(keepdims=True)(inputs)

    # The MLP is shared by average and maximum descriptors, as in CBAM.  A
    # single sigmoid is applied after logits are combined.
    shared_reduce = layers.Conv1D(
        hidden_channels,
        kernel_size=1,
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
    )
    shared_expand = layers.Conv1D(
        channels,
        kernel_size=1,
        use_bias=False,
        kernel_initializer="he_normal",
    )

    average_logits = shared_expand(shared_reduce(average_pool))
    maximum_logits = shared_expand(shared_reduce(maximum_pool))
    return layers.Activation("sigmoid")(
        layers.Add()([average_logits, maximum_logits])
    )


def spatial_attention(inputs, reduction_ratio=8):
    """Estimate attention along the Raman-shift dimension."""
    channels = int(inputs.shape[-1])
    hidden_channels = max(channels // reduction_ratio, 8)

    average_pool = layers.Lambda(
        lambda tensor: tf.reduce_mean(tensor, axis=-1, keepdims=True)
    )(inputs)
    maximum_pool = layers.Lambda(
        lambda tensor: tf.reduce_max(tensor, axis=-1, keepdims=True)
    )(inputs)
    descriptors = layers.Concatenate(axis=-1)([average_pool, maximum_pool])

    attention = layers.Conv1D(
        hidden_channels, kernel_size=3, padding="same", activation="relu"
    )(descriptors)
    attention = layers.Conv1D(
        hidden_channels,
        kernel_size=3,
        dilation_rate=2,
        padding="same",
        activation="relu",
    )(attention)
    return layers.Conv1D(
        1, kernel_size=1, padding="same", activation="sigmoid"
    )(attention)


def corrected_csam_block(inputs, attention_ratio=8):
    """Sequential channel and spectral-spatial attention."""
    channel_weights = channel_attention(inputs, ratio=attention_ratio)
    attended = layers.Multiply()([inputs, channel_weights])
    spatial_weights = spatial_attention(attended, reduction_ratio=attention_ratio)
    return layers.Multiply()([attended, spatial_weights])


def multiscale_residual_block(inputs, num_filters, attention_ratio=8):
    """Residual block covering narrow peaks, broad peaks, and local context."""
    branch_filters = [
        num_filters // 3,
        num_filters // 3,
        num_filters - 2 * (num_filters // 3),
    ]
    branch_specs = ((3, 1), (5, 1), (3, 2))
    branches = []
    for filters, (kernel_size, dilation_rate) in zip(
        branch_filters, branch_specs
    ):
        branch = layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
        branches.append(branch)

    x = layers.Concatenate(axis=-1)(branches)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(
        num_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = corrected_csam_block(x, attention_ratio=attention_ratio)

    if int(inputs.shape[-1]) != num_filters:
        shortcut = layers.Conv1D(
            num_filters, kernel_size=1, padding="same", use_bias=False
        )(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def encoder_block(inputs, num_filters, attention_ratio=8):
    skip = multiscale_residual_block(inputs, num_filters, attention_ratio)
    pooled = layers.MaxPooling1D(pool_size=2, strides=2)(skip)
    return skip, pooled


def attention_gate(skip_tensor, gating_tensor, num_filters):
    """Filter a shallow skip connection using decoder context."""
    skip_projection = layers.Conv1D(
        num_filters, kernel_size=1, padding="same", use_bias=False
    )(skip_tensor)
    gate_projection = layers.Conv1D(
        num_filters, kernel_size=1, padding="same", use_bias=False
    )(gating_tensor)
    gate = layers.Add()([skip_projection, gate_projection])
    gate = layers.Activation("relu")(gate)
    gate = layers.Conv1D(
        1, kernel_size=1, padding="same", activation="sigmoid"
    )(gate)
    return layers.Multiply()([skip_tensor, gate])


def decoder_block(inputs, skip_tensor, num_filters, attention_ratio=8):
    upsampled = layers.Conv1DTranspose(
        num_filters, kernel_size=3, strides=2, padding="same"
    )(inputs)
    gated_skip = attention_gate(skip_tensor, upsampled, num_filters)
    merged = layers.Concatenate(axis=-1)([upsampled, gated_skip])
    return multiscale_residual_block(merged, num_filters, attention_ratio)


def MSAG_CSAM_ResUNet_reconstruction(num_features, num_outputs, args):
    """Build and compile the proposed reconstruction model."""
    base_filters = int(getattr(args, "base_filters", 64))
    attention_ratio = int(getattr(args, "attention_ratio", 8))
    if base_filters < 8:
        raise ValueError("base_filters must be at least 8")
    if attention_ratio < 1:
        raise ValueError("attention_ratio must be positive")

    inputs = layers.Input(shape=(num_features, 1), name="raman_spectrum")
    x1, p1 = encoder_block(inputs, base_filters, attention_ratio)
    x2, p2 = encoder_block(p1, base_filters * 2, attention_ratio)
    x3, p3 = encoder_block(p2, base_filters * 4, attention_ratio)
    x4, p4 = encoder_block(p3, base_filters * 8, attention_ratio)

    bottleneck = multiscale_residual_block(
        p4, base_filters * 16, attention_ratio
    )

    d1 = decoder_block(bottleneck, x4, base_filters * 8, attention_ratio)
    d2 = decoder_block(d1, x3, base_filters * 4, attention_ratio)
    d3 = decoder_block(d2, x2, base_filters * 2, attention_ratio)
    d4 = decoder_block(d3, x1, base_filters, attention_ratio)

    outputs = layers.Conv1D(
        num_outputs,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="reconstructed_spectrum",
    )(d4)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="MSAG_CSAM_ResUNet"
    )

    steps_per_epoch = (
        args.num_spectra
        * float(getattr(args, "train_data_ratio", 0.7))
        / args.batch_size
    )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=0.9772,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule, clipnorm=1.0
    )
    loss = PeakAwareHybridLoss(
        intensity_weight_scale=float(
            getattr(args, "intensity_weight_scale", 100.0)
        ),
        sparse_strength=float(getattr(args, "sparse_loss_weight", 0.005)),
        derivative_strength=float(
            getattr(args, "derivative_loss_weight", 0.05)
        ),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "mae",
            "mape",
            "mse",
            # Axis 1 is the 1024-point spectrum; axis -1 has only one channel.
            CosineSimilarity(axis=1),
            tools.LogCoshError(),
        ],
    )
    model.summary()
    return model


def MSAG_CSAM_ResUNet_unmixing(num_features, num_outputs, args):
    """Build recognition-guided, permutation-invariant spectral unmixing.

    The encoder and the two deepest decoder stages are shared.  The shallow
    decoder stages are component-specific so the model has enough capacity to
    separate overlapping spectra.  Classification consumes only symmetric
    functions of the predicted components, guaranteeing that recognition
    gradients pass through the component outputs while remaining invariant to
    an A/B swap.
    """
    base_filters = int(getattr(args, "base_filters", 64))
    attention_ratio = int(getattr(args, "attention_ratio", 8))
    num_substances = int(getattr(args, "num_substances", 7))
    if base_filters < 8:
        raise ValueError("base_filters must be at least 8")
    if attention_ratio < 1:
        raise ValueError("attention_ratio must be positive")
    if num_features % 16 != 0:
        raise ValueError("num_features must be divisible by 16")
    if int(num_outputs) != 1:
        raise ValueError("unmixing currently requires num_outputs=1")
    if num_substances < 2:
        raise ValueError("num_substances must be at least 2")

    inputs = layers.Input(shape=(num_features, 1), name="raman_spectrum")
    x1, p1 = encoder_block(inputs, base_filters, attention_ratio)
    x2, p2 = encoder_block(p1, base_filters * 2, attention_ratio)
    x3, p3 = encoder_block(p2, base_filters * 4, attention_ratio)
    x4, p4 = encoder_block(p3, base_filters * 8, attention_ratio)
    bottleneck = multiscale_residual_block(
        p4, base_filters * 16, attention_ratio
    )

    shared_d1 = decoder_block(
        bottleneck, x4, base_filters * 8, attention_ratio
    )
    shared_d2 = decoder_block(
        shared_d1, x3, base_filters * 4, attention_ratio
    )

    component_a_d3 = decoder_block(
        shared_d2, x2, base_filters * 2, attention_ratio
    )
    component_a_d4 = decoder_block(
        component_a_d3, x1, base_filters, attention_ratio
    )
    spectrum_a = layers.Conv1D(
        num_outputs,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="spectrum_A",
    )(component_a_d4)

    component_b_d3 = decoder_block(
        shared_d2, x2, base_filters * 2, attention_ratio
    )
    component_b_d4 = decoder_block(
        component_b_d3, x1, base_filters, attention_ratio
    )
    spectrum_b = layers.Conv1D(
        num_outputs,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="spectrum_B",
    )(component_b_d4)

    component_sum = layers.Add(name="component_sum")(
        [spectrum_a, spectrum_b]
    )
    component_difference = AbsoluteDifference(
        name="component_abs_difference"
    )([spectrum_a, spectrum_b])
    classification_features = layers.Concatenate(
        axis=-1, name="recognition_guided_features"
    )([component_sum, component_difference])
    classification_features = layers.Conv1D(
        max(base_filters * 2, 64),
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="classification_branch",
    )(classification_features)
    classification_features = corrected_csam_block(
        classification_features, attention_ratio=attention_ratio
    )
    classification_average = layers.GlobalAveragePooling1D(
        name="classification_average_pool"
    )(classification_features)
    classification_maximum = layers.GlobalMaxPooling1D(
        name="classification_max_pool"
    )(classification_features)
    classification_features = layers.Concatenate(
        name="classification_pool_features"
    )([classification_average, classification_maximum])
    classification_features = layers.Dense(
        256, activation="relu", name="classification_dense"
    )(classification_features)
    classification_features = layers.BatchNormalization(
        name="classification_batch_norm"
    )(classification_features)
    classification_features = layers.Dropout(
        float(getattr(args, "classification_dropout", 0.3)),
        name="classification_dropout",
    )(classification_features)
    classification = layers.Dense(
        num_substances,
        activation="sigmoid",
        name="classification",
    )(classification_features)

    peak_loss_config = {
        "mse_weight": 1.0,
        "intensity_weight_scale": float(
            getattr(args, "peak_weight_scale", 10.0)
        ),
        "intensity_weight_exponent": float(
            getattr(args, "peak_weight_exponent", 2.0)
        ),
        "normalize_intensity": bool(
            getattr(args, "normalize_peak_weights", True)
        ),
        "sparse_strength": float(
            getattr(args, "sparse_loss_weight", 0.002)
        ),
        "derivative_strength": float(
            getattr(args, "derivative_loss_weight", 0.1)
        ),
        "sparse_threshold": float(
            getattr(args, "sparse_threshold", 1e-4)
        ),
    }
    model = RecognitionGuidedUnmixingModel(
        inputs=inputs,
        outputs=[spectrum_a, spectrum_b, classification],
        name="MSAG_CSAM_ResUNet_unmixing",
        peak_loss_config=peak_loss_config,
        unmixing_loss_weight=float(
            getattr(args, "unmixing_loss_weight", 1.0)
        ),
        # BCE is typically orders of magnitude larger than a spectral MSE;
        # start conservatively and ablate 0, .01, .05, and .1.
        classification_loss_weight=float(
            getattr(args, "classification_loss_weight", 0.01)
        ),
        mixture_consistency_weight=float(
            getattr(args, "mixture_consistency_weight", 0.05)
        ),
        classification_label_smoothing=float(
            getattr(args, "classification_label_smoothing", 0.0)
        ),
    )

    steps_per_epoch = (
        args.num_spectra
        * float(getattr(args, "train_data_ratio", 0.7))
        / args.batch_size
    )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=0.9772,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule, clipnorm=1.0
    )
    model.compile(optimizer=optimizer)
    model.summary()
    return model


def MSAG_CSAM_ResUNet_unmixing_only(num_features, num_outputs, args):
    """Build a classification-independent PIT spectral unmixer.

    This network intentionally keeps the same encoder and two-branch decoder
    used by :func:`MSAG_CSAM_ResUNet_unmixing`, but it has no recognition head.
    Consequently, its parameters receive gradients only from the spectral PIT
    objective, mixture consistency, and layer regularization.
    """
    base_filters = int(getattr(args, "base_filters", 64))
    attention_ratio = int(getattr(args, "attention_ratio", 8))
    if base_filters < 8:
        raise ValueError("base_filters must be at least 8")
    if attention_ratio < 1:
        raise ValueError("attention_ratio must be positive")
    if num_features % 16 != 0:
        raise ValueError("num_features must be divisible by 16")
    if int(num_outputs) != 1:
        raise ValueError("unmixing currently requires num_outputs=1")

    inputs = layers.Input(shape=(num_features, 1), name="raman_spectrum")
    x1, p1 = encoder_block(inputs, base_filters, attention_ratio)
    x2, p2 = encoder_block(p1, base_filters * 2, attention_ratio)
    x3, p3 = encoder_block(p2, base_filters * 4, attention_ratio)
    x4, p4 = encoder_block(p3, base_filters * 8, attention_ratio)
    bottleneck = multiscale_residual_block(
        p4, base_filters * 16, attention_ratio
    )

    shared_d1 = decoder_block(
        bottleneck, x4, base_filters * 8, attention_ratio
    )
    shared_d2 = decoder_block(
        shared_d1, x3, base_filters * 4, attention_ratio
    )

    component_a_d3 = decoder_block(
        shared_d2, x2, base_filters * 2, attention_ratio
    )
    component_a_d4 = decoder_block(
        component_a_d3, x1, base_filters, attention_ratio
    )
    spectrum_a = layers.Conv1D(
        num_outputs,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="spectrum_A",
    )(component_a_d4)

    component_b_d3 = decoder_block(
        shared_d2, x2, base_filters * 2, attention_ratio
    )
    component_b_d4 = decoder_block(
        component_b_d3, x1, base_filters, attention_ratio
    )
    spectrum_b = layers.Conv1D(
        num_outputs,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="spectrum_B",
    )(component_b_d4)

    peak_loss_config = {
        "mse_weight": 1.0,
        "intensity_weight_scale": float(
            getattr(args, "peak_weight_scale", 10.0)
        ),
        "intensity_weight_exponent": float(
            getattr(args, "peak_weight_exponent", 2.0)
        ),
        "normalize_intensity": bool(
            getattr(args, "normalize_peak_weights", True)
        ),
        "sparse_strength": float(
            getattr(args, "sparse_loss_weight", 0.002)
        ),
        "derivative_strength": float(
            getattr(args, "derivative_loss_weight", 0.1)
        ),
        "sparse_threshold": float(
            getattr(args, "sparse_threshold", 1e-4)
        ),
    }
    model = PITUnmixingModel(
        inputs=inputs,
        outputs=[spectrum_a, spectrum_b],
        name="MSAG_CSAM_ResUNet_unmixing_only",
        peak_loss_config=peak_loss_config,
        unmixing_loss_weight=float(
            getattr(args, "unmixing_loss_weight", 1.0)
        ),
        mixture_consistency_weight=float(
            getattr(args, "mixture_consistency_weight", 0.05)
        ),
    )

    steps_per_epoch = max(
        1.0,
        args.num_spectra
        * float(getattr(args, "train_data_ratio", 0.7))
        / args.batch_size,
    )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=0.9772,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule, clipnorm=1.0
    )
    model.compile(optimizer=optimizer)
    model.summary()
    return model
