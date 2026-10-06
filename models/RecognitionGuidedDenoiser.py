"""Recognition-guided fine-tuning utilities for Raman denoising.

The inference network is an existing spectral denoiser followed by a small,
zero-initialized residual refinement head.  During fine-tuning, a frozen
classifier supplies label, soft-target, and feature-preservation losses.  The
classifier is deliberately not part of the saved inference model.
"""

import tensorflow as tf
from tensorflow.keras import layers

from models.MSAG_CSAM_ResUNet import PeakAwareHybridLoss


class SpectralSelfSimilarityLoss:
    """Peak-masked local self-similarity loss for one-dimensional spectra.

    The reference spectrum alone defines the peak mask and distance scale, so
    artifacts in the prediction cannot select their own supervision region.
    Local patch relations are compared with symmetric Jensen-Shannon
    divergence plus an L1 probability penalty.
    """

    def __init__(
        self,
        patch_size=7,
        search_radius=15,
        search_step=2,
        peak_fraction=0.25,
        temperature=0.10,
        l1_weight=0.50,
    ):
        self.patch_size = int(patch_size)
        self.search_radius = int(search_radius)
        self.search_step = int(search_step)
        self.peak_fraction = float(peak_fraction)
        self.temperature = float(temperature)
        self.l1_weight = float(l1_weight)
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("self-similarity patch_size must be odd and at least 3")
        if self.search_radius < 1 or self.search_step < 1:
            raise ValueError("self-similarity search sizes must be positive")
        if not 0.0 < self.peak_fraction <= 1.0:
            raise ValueError("self-similarity peak_fraction must be in (0, 1]")
        if self.temperature <= 0 or self.l1_weight < 0:
            raise ValueError("invalid self-similarity loss coefficients")
        self.offsets = tuple(
            offset
            for offset in range(
                -self.search_radius,
                self.search_radius + 1,
                self.search_step,
            )
            if offset != 0
        )
        if not self.offsets:
            raise ValueError("self-similarity search must contain a nonzero offset")

    @staticmethod
    def _flatten_and_normalize(spectra):
        spectra = tf.convert_to_tensor(spectra)
        spectra = tf.reshape(spectra, (tf.shape(spectra)[0], -1))
        minimum = tf.reduce_min(spectra, axis=1, keepdims=True)
        scale = tf.reduce_max(spectra, axis=1, keepdims=True) - minimum
        epsilon = tf.cast(tf.keras.backend.epsilon(), spectra.dtype)
        return (spectra - minimum) / tf.maximum(scale, epsilon)

    def _patches(self, spectra):
        radius = self.patch_size // 2
        padded = tf.pad(spectra, ((0, 0), (radius, radius)), mode="SYMMETRIC")
        return tf.signal.frame(
            padded,
            frame_length=self.patch_size,
            frame_step=1,
            axis=1,
        )

    def _distances(self, patches):
        length = tf.shape(patches)[1]
        centers = tf.range(length)
        distances = []
        valid = []
        for offset in self.offsets:
            shifted = tf.roll(patches, shift=-offset, axis=1)
            distances.append(tf.reduce_mean(tf.square(patches - shifted), axis=-1))
            neighbor = centers + offset
            valid.append(tf.logical_and(neighbor >= 0, neighbor < length))
        return tf.stack(distances, axis=-1), tf.stack(valid, axis=-1)

    def _peak_mask(self, reference):
        first = tf.abs(reference[:, 1:] - reference[:, :-1])
        first = tf.pad(first, ((0, 0), (0, 1)))
        second = tf.abs(
            reference[:, 2:]
            - 2.0 * reference[:, 1:-1]
            + reference[:, :-2]
        )
        second = tf.pad(second, ((0, 0), (1, 1)))
        response = first + second
        ordered = tf.sort(response, axis=-1)
        length = tf.shape(ordered)[1]
        cutoff = tf.cast(
            tf.floor(
                (1.0 - self.peak_fraction)
                * tf.cast(tf.maximum(length - 1, 0), tf.float32)
            ),
            tf.int32,
        )
        threshold = tf.gather(ordered, cutoff, axis=1)
        epsilon = tf.cast(tf.keras.backend.epsilon(), response.dtype)
        mask = tf.logical_and(response >= threshold[:, None], response > epsilon)
        return tf.stop_gradient(tf.cast(mask, response.dtype))

    def per_sample(self, reference, prediction):
        reference = self._flatten_and_normalize(reference)
        prediction = self._flatten_and_normalize(prediction)
        reference_distances, valid = self._distances(self._patches(reference))
        prediction_distances, _ = self._distances(self._patches(prediction))
        valid_float = tf.cast(valid[None, ...], reference.dtype)
        valid_count = tf.reduce_sum(valid_float, axis=-1, keepdims=True)
        local_scale = tf.reduce_sum(
            reference_distances * valid_float, axis=-1, keepdims=True
        ) / tf.maximum(valid_count, tf.cast(1.0, reference.dtype))
        local_scale = tf.stop_gradient(
            tf.maximum(local_scale, tf.cast(1e-4, reference.dtype))
        )
        denominator = tf.cast(self.temperature, reference.dtype) * local_scale
        invalid_logit = tf.cast(-1e4, reference.dtype)
        reference_logits = tf.where(
            valid[None, ...], -reference_distances / denominator, invalid_logit
        )
        prediction_logits = tf.where(
            valid[None, ...], -prediction_distances / denominator, invalid_logit
        )
        reference_probability = tf.nn.softmax(reference_logits, axis=-1)
        prediction_probability = tf.nn.softmax(prediction_logits, axis=-1)
        midpoint = 0.5 * (reference_probability + prediction_probability)
        epsilon = tf.cast(tf.keras.backend.epsilon(), reference.dtype)
        js = 0.5 * tf.reduce_sum(
            reference_probability
            * tf.math.log((reference_probability + epsilon) / (midpoint + epsilon))
            + prediction_probability
            * tf.math.log((prediction_probability + epsilon) / (midpoint + epsilon)),
            axis=-1,
        )
        probability_l1 = tf.reduce_mean(
            tf.abs(reference_probability - prediction_probability), axis=-1
        )
        point_loss = js + tf.cast(self.l1_weight, reference.dtype) * probability_l1
        peak_mask = self._peak_mask(reference)
        peak_count = tf.reduce_sum(peak_mask, axis=-1)
        loss = tf.reduce_sum(point_loss * peak_mask, axis=-1) / tf.maximum(
            peak_count, tf.cast(1.0, reference.dtype)
        )
        mask_fraction = tf.reduce_mean(peak_mask, axis=-1)
        return loss, mask_fraction


@tf.keras.utils.register_keras_serializable(package="Raman")
class SpectralQuality(layers.Layer):
    """Per-spectrum noise proxy; no labels or batch statistics are used."""

    def call(self, spectrum):
        x = spectrum[..., 0]
        second = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        ordered = tf.sort(tf.abs(second), axis=-1)
        median = ordered[:, tf.shape(ordered)[1] // 2]
        noise = median / (0.67448975 * tf.sqrt(6.0))
        amplitude = tf.reduce_max(x, axis=-1) - tf.reduce_min(x, axis=-1)
        relative = noise / tf.maximum(amplitude, 1e-6)
        return tf.stack([tf.math.log1p(noise), tf.math.log1p(relative)], -1)


def build_recognition_guided_denoiser(
    base_denoiser,
    num_features=1024,
    refinement_filters=32,
    maximum_correction=0.10,
    quality_conditioning=False,
):
    """Wrap a pretrained denoiser with a zero-initialized refinement head.

    At initialization ``recognition_guided_spectrum`` is exactly the output of
    ``base_denoiser``.  This property makes recognition fine-tuning safe to
    start: the new head cannot immediately destroy a good reconstruction.
    """
    refinement_filters = int(refinement_filters)
    if refinement_filters < 6:
        raise ValueError("refinement_filters must be at least 6")
    if maximum_correction <= 0:
        raise ValueError("maximum_correction must be positive")

    inputs = layers.Input(
        shape=(int(num_features), 1), name="noisy_raman_spectrum"
    )
    base_spectrum = base_denoiser(inputs)
    if isinstance(base_spectrum, (tuple, list)):
        raise ValueError("base_denoiser must have exactly one spectral output")

    absolute_residual = layers.Lambda(
        lambda values: tf.abs(values[0] - values[1]),
        name="absolute_input_residual",
    )([inputs, base_spectrum])
    refinement_input = layers.Concatenate(
        axis=-1, name="refinement_input"
    )([inputs, base_spectrum, absolute_residual])

    branch_widths = (
        refinement_filters // 3,
        refinement_filters // 3,
        refinement_filters - 2 * (refinement_filters // 3),
    )
    branches = []
    for branch_index, (width, kernel_size) in enumerate(
        zip(branch_widths, (3, 7, 15)), start=1
    ):
        branch = layers.Conv1D(
            width,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"refinement_branch_{branch_index}",
        )(refinement_input)
        branches.append(branch)

    refinement = layers.Concatenate(
        axis=-1, name="multiscale_refinement"
    )(branches)
    refinement = layers.BatchNormalization(
        name="refinement_batch_norm"
    )(refinement)
    refinement = layers.Activation(
        "relu", name="refinement_activation"
    )(refinement)
    refinement = layers.Conv1D(
        refinement_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="refinement_context",
    )(refinement)
    model_inputs = inputs
    if quality_conditioning:
        # log1p(power_mW)/log1p(150), log1p(time_ms)/log1p(5000),
        # metadata-valid flag. Unknown/synthetic acquisition uses [0,0,0].
        acquisition = layers.Input((3,), name="acquisition_quality")
        quality = layers.Concatenate(name="quality_features")([
            acquisition, SpectralQuality(name="spectral_quality")(inputs)
        ])
        quality = layers.Dense(16, activation="relu", name="quality_embedding")(quality)
        quality = layers.RepeatVector(int(num_features), name="quality_broadcast")(quality)
        gate_input = layers.Concatenate(name="quality_gate_input")([refinement, quality])
        model_inputs = [inputs, acquisition]
    else:
        gate_input = refinement
    gate = layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
        name="recognition_refinement_gate",
    )(gate_input)
    correction = layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        activation="tanh",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="recognition_correction",
    )(refinement)
    correction = layers.Multiply(
        name="gated_recognition_correction"
    )([gate, correction])
    correction = layers.Lambda(
        lambda value: value * tf.cast(maximum_correction, value.dtype),
        name="bounded_recognition_correction",
    )(correction)
    outputs = layers.Add(name="base_plus_recognition_correction")(
        [base_spectrum, correction]
    )
    outputs = layers.ReLU(name="recognition_guided_spectrum")(outputs)

    return tf.keras.Model(
        inputs=model_inputs,
        outputs=outputs,
        name="RecognitionGuidedDenoiser",
    )


class RamanNoiseAugmenter(layers.Layer):
    """Generate low-SNR Raman-like inputs from clean simulated mixtures."""

    def __init__(
        self,
        gaussian_std_min=0.002,
        gaussian_std_max=0.035,
        shot_noise_max=0.025,
        baseline_max=0.20,
        spike_probability=0.001,
        spike_amplitude=0.20,
        gain_min=0.85,
        gain_max=1.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gaussian_std_min = float(gaussian_std_min)
        self.gaussian_std_max = float(gaussian_std_max)
        self.shot_noise_max = float(shot_noise_max)
        self.baseline_max = float(baseline_max)
        self.spike_probability = float(spike_probability)
        self.spike_amplitude = float(spike_amplitude)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        if not 0 <= self.gaussian_std_min <= self.gaussian_std_max:
            raise ValueError("invalid Gaussian-noise range")
        if not 0 <= self.gain_min <= self.gain_max:
            raise ValueError("invalid gain range")
        if min(
            self.shot_noise_max,
            self.baseline_max,
            self.spike_probability,
            self.spike_amplitude,
        ) < 0:
            raise ValueError("noise strengths must be nonnegative")

    @staticmethod
    def _uniform(batch_size, minimum, maximum, dtype):
        return tf.random.uniform(
            (batch_size, 1, 1), minimum, maximum, dtype=dtype
        )

    def call(self, clean_spectra, training=None):
        if training is False:
            return clean_spectra
        clean_spectra = tf.convert_to_tensor(clean_spectra)
        dtype = clean_spectra.dtype
        shape = tf.shape(clean_spectra)
        batch_size = shape[0]
        length = shape[1]
        epsilon = tf.cast(tf.keras.backend.epsilon(), dtype)

        gain = self._uniform(
            batch_size, self.gain_min, self.gain_max, dtype
        )
        degraded = gain * clean_spectra

        gaussian_std = self._uniform(
            batch_size,
            self.gaussian_std_min,
            self.gaussian_std_max,
            dtype,
        )
        degraded += tf.random.normal(shape, dtype=dtype) * gaussian_std

        signal_scale = tf.reduce_max(
            tf.maximum(clean_spectra, 0.0), axis=1, keepdims=True
        )
        relative_signal = tf.maximum(clean_spectra, 0.0) / (
            signal_scale + epsilon
        )
        shot_strength = self._uniform(
            batch_size, 0.0, self.shot_noise_max, dtype
        )
        degraded += (
            tf.random.normal(shape, dtype=dtype)
            * shot_strength
            * tf.sqrt(relative_signal + epsilon)
        )

        coordinate = tf.reshape(
            tf.linspace(tf.cast(-1.0, dtype), tf.cast(1.0, dtype), length),
            (1, length, 1),
        )
        baseline_amplitude = self._uniform(
            batch_size, 0.0, self.baseline_max, dtype
        )
        baseline_center = self._uniform(batch_size, -0.8, 0.8, dtype)
        baseline_curve = self._uniform(batch_size, 0.2, 1.0, dtype)
        baseline = baseline_amplitude * (
            0.25 + baseline_curve * tf.square(coordinate - baseline_center)
        )
        degraded += baseline

        if self.spike_probability > 0 and self.spike_amplitude > 0:
            spike_mask = tf.cast(
                tf.random.uniform(shape, dtype=dtype)
                < self.spike_probability,
                dtype,
            )
            spike_height = tf.random.uniform(
                shape,
                0.0,
                self.spike_amplitude,
                dtype=dtype,
            )
            degraded += spike_mask * spike_height

        degraded = tf.maximum(degraded, 0.0)
        maximum = tf.reduce_max(degraded, axis=1, keepdims=True)
        return degraded / (maximum + epsilon)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gaussian_std_min": self.gaussian_std_min,
                "gaussian_std_max": self.gaussian_std_max,
                "shot_noise_max": self.shot_noise_max,
                "baseline_max": self.baseline_max,
                "spike_probability": self.spike_probability,
                "spike_amplitude": self.spike_amplitude,
                "gain_min": self.gain_min,
                "gain_max": self.gain_max,
            }
        )
        return config


class RecognitionGuidedTrainingModel(tf.keras.Model):
    """Train a denoiser through a frozen classification feature extractor."""

    def __init__(
        self,
        denoiser,
        teacher,
        augmenter,
        reconstruction_loss_config=None,
        reconstruction_weight=1.0,
        classification_weight=0.001,
        distillation_weight=0.001,
        feature_weight=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        self.teacher = teacher
        self.teacher.trainable = False
        self.augmenter = augmenter
        self.reconstruction_weight = float(reconstruction_weight)
        self.classification_weight = float(classification_weight)
        self.distillation_weight = float(distillation_weight)
        self.feature_weight = float(feature_weight)
        self.reconstruction_loss_fn = PeakAwareHybridLoss(
            **dict(reconstruction_loss_config or {})
        )
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        metric_names = (
            "loss",
            "reconstruction_loss",
            "classification_loss",
            "distillation_loss",
            "feature_loss",
            "guided_subset_accuracy",
            "teacher_clean_subset_accuracy",
        )
        self.trackers = {
            name: tf.keras.metrics.Mean(name=name) for name in metric_names
        }
        self.spectral_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectral_mae"
        )

    @property
    def metrics(self):
        return list(self.trackers.values()) + [self.spectral_mae]

    def call(self, inputs, training=False):
        return self.denoiser(inputs, training=training)

    @staticmethod
    def _unpack_targets(targets):
        if not isinstance(targets, dict):
            raise ValueError(
                "targets must contain clean_spectrum and classification"
            )
        return targets["clean_spectrum"], targets["classification"]

    @staticmethod
    def _top_two_subset_accuracy(labels, probabilities):
        top_indices = tf.math.top_k(probabilities, k=2).indices
        predicted = tf.reduce_sum(
            tf.one_hot(
                top_indices,
                depth=tf.shape(probabilities)[-1],
                dtype=labels.dtype,
            ),
            axis=1,
        )
        exact = tf.reduce_all(tf.equal(labels, predicted), axis=1)
        return tf.reduce_mean(tf.cast(exact, tf.float32))

    def _calculate_losses(self, noisy_spectra, clean_spectra, labels, training):
        denoised = self.denoiser(noisy_spectra, training=training)
        clean_features, clean_probabilities = self.teacher(
            clean_spectra, training=False
        )
        guided_features, guided_probabilities = self.teacher(
            denoised, training=False
        )
        clean_features = tf.stop_gradient(clean_features)
        clean_probabilities = tf.stop_gradient(clean_probabilities)

        reconstruction_loss = tf.reduce_mean(
            self.reconstruction_loss_fn.per_sample(clean_spectra, denoised)
        )
        classification_loss = tf.reduce_mean(
            self.binary_crossentropy(labels, guided_probabilities)
        )
        distillation_loss = tf.reduce_mean(
            self.binary_crossentropy(
                clean_probabilities, guided_probabilities
            )
        )
        clean_features = tf.math.l2_normalize(clean_features, axis=-1)
        guided_features = tf.math.l2_normalize(guided_features, axis=-1)
        feature_loss = tf.reduce_mean(
            1.0 - tf.reduce_sum(clean_features * guided_features, axis=-1)
        )

        regularization_loss = tf.cast(0.0, reconstruction_loss.dtype)
        if self.denoiser.losses:
            regularization_loss = tf.add_n(self.denoiser.losses)
        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.classification_weight * classification_loss
            + self.distillation_weight * distillation_loss
            + self.feature_weight * feature_loss
            + regularization_loss
        )
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
            "guided_subset_accuracy": self._top_two_subset_accuracy(
                labels, guided_probabilities
            ),
            "teacher_clean_subset_accuracy": self._top_two_subset_accuracy(
                labels, clean_probabilities
            ),
            "denoised": denoised,
        }

    def _update_metrics(self, values, clean_spectra):
        for name, tracker in self.trackers.items():
            tracker.update_state(values[name])
        self.spectral_mae.update_state(clean_spectra, values["denoised"])

    def train_step(self, data):
        clean_spectra, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        target_spectra, labels = self._unpack_targets(targets)
        noisy_spectra = self.augmenter(clean_spectra, training=True)
        with tf.GradientTape() as tape:
            values = self._calculate_losses(
                noisy_spectra, target_spectra, labels, training=True
            )
        gradients = tape.gradient(
            values["loss"], self.denoiser.trainable_variables
        )
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.denoiser.trainable_variables
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        self._update_metrics(values, target_spectra)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        noisy_spectra, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        target_spectra, labels = self._unpack_targets(targets)
        values = self._calculate_losses(
            noisy_spectra, target_spectra, labels, training=False
        )
        self._update_metrics(values, target_spectra)
        return {metric.name: metric.result() for metric in self.metrics}


class RealDomainAdaptationModel(tf.keras.Model):
    """Adapt a denoiser with paired raw and conventionally denoised spectra.

    The recognition teacher remains frozen.  Teacher features and probabilities
    for the target spectra can be precomputed and supplied in ``targets`` to
    avoid a second teacher forward pass in every training step.
    """

    def __init__(
        self,
        denoiser,
        teacher,
        reconstruction_loss_config=None,
        reconstruction_weight=1.0,
        classification_weight=0.0005,
        distillation_weight=0.002,
        feature_weight=0.002,
        sam_weight=0.05,
        correction_weight=0.01,
        safe_margin_weight=0.01,
        self_similarity_weight=0.0,
        self_similarity_loss_config=None,
        confidence_margin=0.20,
        hard_example_weight=3.0,
        easy_identity_weight=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        self.teacher = teacher
        self.teacher.trainable = False
        self.reconstruction_weight = float(reconstruction_weight)
        self.classification_weight = float(classification_weight)
        self.distillation_weight = float(distillation_weight)
        self.feature_weight = float(feature_weight)
        self.sam_weight = float(sam_weight)
        self.correction_weight = float(correction_weight)
        self.safe_margin_weight = float(safe_margin_weight)
        self.self_similarity_weight = float(self_similarity_weight)
        self.confidence_margin = float(confidence_margin)
        self.hard_example_weight = float(hard_example_weight)
        self.easy_identity_weight = float(easy_identity_weight)
        self.reconstruction_loss_fn = PeakAwareHybridLoss(
            **dict(reconstruction_loss_config or {})
        )
        self.self_similarity_loss_fn = SpectralSelfSimilarityLoss(
            **dict(self_similarity_loss_config or {})
        )
        if self.self_similarity_weight < 0:
            raise ValueError("self_similarity_weight must be nonnegative")
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        try:
            correction = denoiser.get_layer(
                "bounded_recognition_correction"
            ).output
        except ValueError:
            correction = layers.Lambda(
                lambda value: tf.zeros_like(value),
                name="zero_correction_for_loaded_model",
            )(denoiser.output)
        self.inference_with_correction = tf.keras.Model(
            denoiser.input,
            [denoiser.output, correction],
            name="DenoiserWithCorrectionProbe",
        )

        metric_names = (
            "loss",
            "reconstruction_loss",
            "classification_loss",
            "distillation_loss",
            "feature_loss",
            "sam_loss",
            "correction_loss",
            "safe_margin_loss",
            "self_similarity_loss",
            "peak_mask_fraction",
            "guided_subset_accuracy",
            "target_subset_accuracy",
            "easy_sample_fraction",
            "margin_violation_rate",
            "target_class_margin",
            "guided_class_margin",
            "sam_degrees",
        )
        self.trackers = {
            name: tf.keras.metrics.Mean(name=name) for name in metric_names
        }
        self.spectral_mae = tf.keras.metrics.MeanAbsoluteError(
            name="spectral_mae"
        )

    @property
    def metrics(self):
        return list(self.trackers.values()) + [self.spectral_mae]

    def call(self, inputs, training=False):
        return self.denoiser(inputs, training=training)

    @staticmethod
    def _unpack_targets(targets):
        if not isinstance(targets, dict):
            raise ValueError(
                "targets must contain target_spectrum and classification"
            )
        required = (
            "target_spectrum",
            "classification",
            "teacher_features",
            "teacher_probabilities",
        )
        missing = [key for key in required if key not in targets]
        if missing:
            raise ValueError("missing target fields: " + ", ".join(missing))
        return tuple(targets[key] for key in required)

    @staticmethod
    def _top_two_exact(labels, probabilities):
        top_indices = tf.math.top_k(probabilities, k=2).indices
        predicted = tf.reduce_sum(
            tf.one_hot(
                top_indices,
                depth=tf.shape(probabilities)[-1],
                dtype=labels.dtype,
            ),
            axis=1,
        )
        exact = tf.reduce_all(tf.equal(labels, predicted), axis=1)
        return tf.cast(exact, tf.float32)

    @classmethod
    def _top_two_subset_accuracy(cls, labels, probabilities):
        return tf.reduce_mean(cls._top_two_exact(labels, probabilities))

    @staticmethod
    def _classification_margin(labels, probabilities):
        labels = tf.cast(labels, probabilities.dtype)
        large = tf.cast(1e6, probabilities.dtype)
        smallest_true = tf.reduce_min(
            tf.where(labels > 0.5, probabilities, large), axis=-1
        )
        largest_false = tf.reduce_max(
            tf.where(labels <= 0.5, probabilities, -large), axis=-1
        )
        return smallest_true - largest_false

    @staticmethod
    def _weighted_mean(values, weights):
        values = tf.cast(values, weights.dtype)
        epsilon = tf.cast(tf.keras.backend.epsilon(), weights.dtype)
        return tf.reduce_sum(values * weights) / (
            tf.reduce_sum(weights) + epsilon
        )

    @staticmethod
    def _cosine_distance(reference, prediction):
        reference = tf.reshape(reference, (tf.shape(reference)[0], -1))
        prediction = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
        reference = tf.math.l2_normalize(reference, axis=-1)
        prediction = tf.math.l2_normalize(prediction, axis=-1)
        cosine = tf.reduce_sum(reference * prediction, axis=-1)
        return 1.0 - tf.clip_by_value(cosine, -1.0, 1.0), cosine

    def _calculate_losses(
        self,
        raw_spectra,
        target_spectra,
        labels,
        target_features,
        target_probabilities,
        training,
    ):
        denoised, correction = self.inference_with_correction(
            raw_spectra, training=training
        )
        guided_features, guided_probabilities = self.teacher(
            denoised, training=False
        )
        target_features = tf.stop_gradient(target_features)
        target_probabilities = tf.stop_gradient(target_probabilities)

        target_exact = self._top_two_exact(labels, target_probabilities)
        target_margin = self._classification_margin(
            labels, target_probabilities
        )
        guided_margin = self._classification_margin(
            labels, guided_probabilities
        )
        easy_mask = tf.cast(
            tf.logical_and(
                target_exact > 0.5,
                target_margin >= tf.cast(
                    self.confidence_margin, target_margin.dtype
                ),
            ),
            target_margin.dtype,
        )
        hard_weights = 1.0 + self.hard_example_weight * (1.0 - easy_mask)
        identity_weights = 1.0 + self.easy_identity_weight * easy_mask

        reconstruction_per_sample = self.reconstruction_loss_fn.per_sample(
            target_spectra, denoised
        )
        reconstruction_loss = self._weighted_mean(
            reconstruction_per_sample, identity_weights
        )
        classification_per_sample = self.binary_crossentropy(
            labels, guided_probabilities
        )
        classification_loss = self._weighted_mean(
            classification_per_sample, hard_weights
        )
        distillation_loss = tf.reduce_mean(
            self.binary_crossentropy(
                target_probabilities, guided_probabilities
            )
        )
        target_features = tf.math.l2_normalize(target_features, axis=-1)
        guided_features = tf.math.l2_normalize(guided_features, axis=-1)
        feature_loss = tf.reduce_mean(
            1.0
            - tf.reduce_sum(target_features * guided_features, axis=-1)
        )
        sam_per_sample, cosine = self._cosine_distance(
            target_spectra, denoised
        )
        sam_loss = tf.reduce_mean(sam_per_sample)
        safe_cosine = tf.clip_by_value(cosine, -0.999999, 0.999999)
        sam_degrees = tf.reduce_mean(
            tf.acos(safe_cosine) * tf.cast(180.0 / 3.141592653589793, cosine.dtype)
        )
        correction_per_sample = tf.reduce_mean(
            tf.abs(correction), axis=[1, 2]
        )
        correction_loss = self._weighted_mean(
            correction_per_sample, identity_weights
        )
        margin_shortfall = tf.nn.relu(target_margin - guided_margin)
        safe_margin_loss = self._weighted_mean(
            margin_shortfall, identity_weights
        )
        margin_violation_rate = tf.reduce_mean(
            tf.cast(margin_shortfall > 1e-6, tf.float32)
        )

        if self.self_similarity_weight > 0:
            self_similarity_per_sample, peak_mask_fraction = (
                self.self_similarity_loss_fn.per_sample(
                    target_spectra, denoised
                )
            )
            self_similarity_loss = tf.reduce_mean(
                self_similarity_per_sample
            )
            peak_mask_fraction = tf.reduce_mean(peak_mask_fraction)
        else:
            self_similarity_loss = tf.cast(0.0, reconstruction_loss.dtype)
            peak_mask_fraction = tf.cast(0.0, reconstruction_loss.dtype)

        regularization_loss = tf.cast(0.0, reconstruction_loss.dtype)
        if self.denoiser.losses:
            regularization_loss = tf.add_n(self.denoiser.losses)
        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.classification_weight * classification_loss
            + self.distillation_weight * distillation_loss
            + self.feature_weight * feature_loss
            + self.sam_weight * sam_loss
            + self.correction_weight * correction_loss
            + self.safe_margin_weight * safe_margin_loss
            + self.self_similarity_weight * self_similarity_loss
            + regularization_loss
        )
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
            "sam_loss": sam_loss,
            "correction_loss": correction_loss,
            "safe_margin_loss": safe_margin_loss,
            "self_similarity_loss": self_similarity_loss,
            "peak_mask_fraction": peak_mask_fraction,
            "guided_subset_accuracy": self._top_two_subset_accuracy(
                labels, guided_probabilities
            ),
            "target_subset_accuracy": self._top_two_subset_accuracy(
                labels, target_probabilities
            ),
            "easy_sample_fraction": tf.reduce_mean(easy_mask),
            "margin_violation_rate": margin_violation_rate,
            "target_class_margin": tf.reduce_mean(target_margin),
            "guided_class_margin": tf.reduce_mean(guided_margin),
            "sam_degrees": sam_degrees,
            "denoised": denoised,
            # Available to derived adaptation models without changing the
            # legacy metric list.
            "guided_features": guided_features,
            "guided_probabilities": guided_probabilities,
        }

    def _step(self, data, training):
        raw_spectra, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        (
            target_spectra,
            labels,
            target_features,
            target_probabilities,
        ) = self._unpack_targets(targets)

        if training:
            with tf.GradientTape() as tape:
                values = self._calculate_losses(
                    raw_spectra,
                    target_spectra,
                    labels,
                    target_features,
                    target_probabilities,
                    training=True,
                )
            gradients = tape.gradient(
                values["loss"], self.denoiser.trainable_variables
            )
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(
                    gradients, self.denoiser.trainable_variables
                )
                if gradient is not None
            ]
            self.optimizer.apply_gradients(pairs)
        else:
            values = self._calculate_losses(
                raw_spectra,
                target_spectra,
                labels,
                target_features,
                target_probabilities,
                training=False,
            )

        for name, tracker in self.trackers.items():
            tracker.update_state(values[name])
        self.spectral_mae.update_state(target_spectra, values["denoised"])
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)


class _MultiQualityUnpairedMixin:
    """Adapt with multi-condition spectra without assuming row correspondence.

    A high-quality reference from the same material-pair/particle-size group is
    degraded online, creating a valid synthetic input/target pair.  Real
    spectra are aligned only to high-quality group moments; equally numbered
    rows from different acquisition conditions are never compared directly.
    """

    def __init__(
        self,
        denoiser,
        teacher,
        unmixer,
        reference_centroids,
        high_reference_raw,
        high_reference_target,
        high_reference_labels,
        high_reference_probabilities,
        group_spectral_mean,
        group_spectral_std,
        group_feature_mean,
        group_feature_std,
        synthetic_pair_weight=0.02,
        synthetic_classification_weight=0.001,
        high_quality_identity_weight=1.0,
        group_spectral_weight=0.05,
        group_feature_weight=0.02,
        synthetic_self_similarity_weight=0.0,
        augmentation_seed=19580920,
        gaussian_std_min=0.001,
        gaussian_std_max=0.030,
        shot_noise_max=0.020,
        baseline_max=0.120,
        spike_probability=0.0005,
        spike_amplitude=0.150,
        gain_variation=0.12,
        high_reference_quality=(0.0, 0.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            denoiser=denoiser,
            teacher=teacher,
            unmixer=unmixer,
            reference_centroids=reference_centroids,
            **kwargs,
        )
        self.high_reference_raw = tf.convert_to_tensor(
            high_reference_raw, dtype=tf.float32
        )
        self.high_reference_quality = tf.convert_to_tensor(high_reference_quality, tf.float32)
        self.high_reference_target = tf.convert_to_tensor(
            high_reference_target, dtype=tf.float32
        )
        self.high_reference_labels = tf.convert_to_tensor(
            high_reference_labels, dtype=tf.float32
        )
        self.high_reference_probabilities = tf.convert_to_tensor(
            high_reference_probabilities, dtype=tf.float32
        )
        self.group_spectral_mean = tf.convert_to_tensor(
            group_spectral_mean, dtype=tf.float32
        )
        self.group_spectral_std = tf.convert_to_tensor(
            group_spectral_std, dtype=tf.float32
        )
        self.group_feature_mean = tf.convert_to_tensor(
            group_feature_mean, dtype=tf.float32
        )
        self.group_feature_std = tf.convert_to_tensor(
            group_feature_std, dtype=tf.float32
        )
        self.synthetic_pair_weight = float(synthetic_pair_weight)
        self.synthetic_classification_weight = float(
            synthetic_classification_weight
        )
        self.high_quality_identity_weight = float(
            high_quality_identity_weight
        )
        self.group_spectral_weight = float(group_spectral_weight)
        self.group_feature_weight = float(group_feature_weight)
        self.synthetic_self_similarity_weight = float(
            synthetic_self_similarity_weight
        )
        self.augmentation_seed = int(augmentation_seed)
        self.gaussian_std_min = float(gaussian_std_min)
        self.gaussian_std_max = float(gaussian_std_max)
        self.shot_noise_max = float(shot_noise_max)
        self.baseline_max = float(baseline_max)
        self.spike_probability = float(spike_probability)
        self.spike_amplitude = float(spike_amplitude)
        self.gain_variation = float(gain_variation)

        nonnegative = (
            self.synthetic_pair_weight,
            self.synthetic_classification_weight,
            self.high_quality_identity_weight,
            self.group_spectral_weight,
            self.group_feature_weight,
            self.synthetic_self_similarity_weight,
            self.gaussian_std_min,
            self.gaussian_std_max,
            self.shot_noise_max,
            self.baseline_max,
            self.spike_probability,
            self.spike_amplitude,
            self.gain_variation,
        )
        if min(nonnegative) < 0:
            raise ValueError("multi-quality weights must be nonnegative")
        if self.gaussian_std_min > self.gaussian_std_max:
            raise ValueError("invalid Gaussian-noise range")

        for name in (
            "synthetic_pair_loss",
            "synthetic_classification_loss",
            "high_quality_identity_loss",
            "group_spectral_loss",
            "group_feature_loss",
            "synthetic_sam_degrees",
            "synthetic_self_similarity_loss",
        ):
            self.trackers[name] = tf.keras.metrics.Mean(name=name)

    @staticmethod
    def _required_unpaired_targets(targets):
        required = ("group_id", "reference_index", "augmentation_index")
        missing = [key for key in required if key not in targets]
        if missing:
            raise ValueError("missing target fields: " + ", ".join(missing))
        return tuple(targets[key] for key in required)

    @staticmethod
    def _moment_loss(values, group_ids, target_mean, target_std):
        """Match means/stds for each group represented in the mini-batch."""
        values = tf.reshape(values, (tf.shape(values)[0], -1))
        target_mean = tf.reshape(target_mean, (tf.shape(target_mean)[0], -1))
        target_std = tf.reshape(target_std, (tf.shape(target_std)[0], -1))
        group_ids = tf.cast(tf.reshape(group_ids, (-1,)), tf.int32)
        group_count = tf.shape(target_mean)[0]
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(group_ids, dtype=values.dtype),
            group_ids,
            group_count,
        )
        sums = tf.math.unsorted_segment_sum(values, group_ids, group_count)
        safe_counts = tf.maximum(counts, tf.cast(1.0, values.dtype))
        means = sums / safe_counts[:, None]
        centered = values - tf.gather(means, group_ids)
        variances = tf.math.unsorted_segment_sum(
            tf.square(centered), group_ids, group_count
        ) / safe_counts[:, None]
        stds = tf.sqrt(variances + tf.cast(1e-8, values.dtype))

        represented = tf.cast(counts > 0, values.dtype)[:, None]
        dimensions = tf.cast(tf.shape(values)[1], values.dtype)
        mean_denominator = tf.reduce_sum(represented) * dimensions
        mean_error = tf.reduce_sum(
            represented * tf.square(means - target_mean)
        ) / tf.maximum(mean_denominator, tf.cast(1.0, values.dtype))

        # A singleton does not provide a within-batch variance estimate.
        variance_observed = tf.cast(counts > 1, values.dtype)[:, None]
        std_denominator = tf.reduce_sum(variance_observed) * dimensions
        std_error = tf.reduce_sum(
            variance_observed * tf.square(stds - target_std)
        ) / tf.maximum(std_denominator, tf.cast(1.0, values.dtype))
        return mean_error + std_error

    def _batch_seed(self, augmentation_index, offset):
        value = tf.reduce_sum(tf.cast(augmentation_index, tf.int64))
        value = tf.cast(
            tf.math.floormod(value, tf.cast(2147483000, tf.int64)),
            tf.int32,
        )
        return tf.stack(
            (tf.cast(self.augmentation_seed + offset, tf.int32), value)
        )

    def _degrade(self, clean, severity, augmentation_index):
        """Apply condition-scaled Raman-like corruption reproducibly."""
        clean = tf.convert_to_tensor(clean)
        dtype = clean.dtype
        shape = tf.shape(clean)
        batch_size = shape[0]
        length = shape[1]
        severity = tf.cast(tf.reshape(severity, (-1, 1, 1)), dtype)
        epsilon = tf.cast(tf.keras.backend.epsilon(), dtype)

        gain_noise = tf.random.stateless_uniform(
            (batch_size, 1, 1),
            self._batch_seed(augmentation_index, 1),
            minval=-1.0,
            maxval=1.0,
            dtype=dtype,
        )
        degraded = clean * (
            1.0 + severity * self.gain_variation * gain_noise
        )
        gaussian_std = self.gaussian_std_min + severity * (
            self.gaussian_std_max - self.gaussian_std_min
        )
        degraded += tf.random.stateless_normal(
            shape,
            self._batch_seed(augmentation_index, 2),
            dtype=dtype,
        ) * gaussian_std

        positive = tf.maximum(clean, 0.0)
        scale = tf.reduce_max(positive, axis=1, keepdims=True)
        relative = positive / (scale + epsilon)
        degraded += (
            tf.random.stateless_normal(
                shape,
                self._batch_seed(augmentation_index, 3),
                dtype=dtype,
            )
            * severity
            * self.shot_noise_max
            * tf.sqrt(relative + epsilon)
        )

        coordinate = tf.reshape(
            tf.linspace(tf.cast(-1.0, dtype), tf.cast(1.0, dtype), length),
            (1, length, 1),
        )
        center = tf.random.stateless_uniform(
            (batch_size, 1, 1),
            self._batch_seed(augmentation_index, 4),
            minval=-0.8,
            maxval=0.8,
            dtype=dtype,
        )
        curve = tf.random.stateless_uniform(
            (batch_size, 1, 1),
            self._batch_seed(augmentation_index, 5),
            minval=0.2,
            maxval=1.0,
            dtype=dtype,
        )
        amplitude = tf.random.stateless_uniform(
            (batch_size, 1, 1),
            self._batch_seed(augmentation_index, 6),
            minval=0.0,
            maxval=self.baseline_max,
            dtype=dtype,
        )
        degraded += severity * amplitude * (
            0.25 + curve * tf.square(coordinate - center)
        )

        if self.spike_probability > 0 and self.spike_amplitude > 0:
            spike_probability = severity * self.spike_probability
            spike_mask = tf.cast(
                tf.random.stateless_uniform(
                    shape,
                    self._batch_seed(augmentation_index, 7),
                    dtype=dtype,
                )
                < spike_probability,
                dtype,
            )
            spike_height = tf.random.stateless_uniform(
                shape,
                self._batch_seed(augmentation_index, 8),
                minval=0.0,
                maxval=self.spike_amplitude,
                dtype=dtype,
            )
            degraded += spike_mask * spike_height

        degraded = tf.maximum(degraded, 0.0)
        maximum = tf.reduce_max(degraded, axis=1, keepdims=True)
        return degraded / (maximum + epsilon)

    def _unpaired_values(
        self,
        values,
        group_ids,
        reference_indices,
        augmentation_index,
        training,
    ):
        reference_indices = tf.cast(
            tf.reshape(reference_indices, (-1,)), tf.int32
        )
        high_raw = tf.gather(self.high_reference_raw, reference_indices)
        high_target = tf.gather(
            self.high_reference_target, reference_indices
        )
        high_labels = tf.gather(
            self.high_reference_labels, reference_indices
        )
        high_probabilities = tf.gather(
            self.high_reference_probabilities, reference_indices
        )
        augmentation_index = tf.cast(augmentation_index, tf.int32)
        severity = tf.cast(augmentation_index[:, 1], tf.float32) / tf.cast(
            1000000.0, tf.float32
        )
        synthetic_input = self._degrade(
            high_target, severity, augmentation_index[:, 0]
        )
        identity_input = high_raw
        if len(self.denoiser.inputs) == 2:
            batch = tf.shape(high_raw)[0]
            # Synthetic corruption has no genuine acquisition metadata.
            synthetic_input = (synthetic_input, tf.zeros((batch, 3), tf.float32))
            identity_input = (high_raw, tf.tile(self.high_reference_quality[None, :], (batch, 1)))
        synthetic_output = self.denoiser(
            synthetic_input, training=training
        )
        identity_output = self.denoiser(identity_input, training=training)

        synthetic_pair_loss = tf.reduce_mean(
            self.reconstruction_loss_fn.per_sample(
                high_target, synthetic_output
            )
        )
        high_quality_identity_loss = tf.reduce_mean(
            self.reconstruction_loss_fn.per_sample(
                high_target, identity_output
            )
        )
        _, synthetic_probabilities = self.teacher(
            synthetic_output, training=False
        )
        synthetic_classification_loss = 0.5 * (
            tf.reduce_mean(
                self.binary_crossentropy(
                    high_labels, synthetic_probabilities
                )
            )
            + tf.reduce_mean(
                self.binary_crossentropy(
                    high_probabilities, synthetic_probabilities
                )
            )
        )

        normalized_features = tf.math.l2_normalize(
            values["guided_features"], axis=-1
        )
        group_spectral_loss = self._moment_loss(
            values["denoised"],
            group_ids,
            self.group_spectral_mean,
            self.group_spectral_std,
        )
        group_feature_loss = self._moment_loss(
            normalized_features,
            group_ids,
            self.group_feature_mean,
            self.group_feature_std,
        )
        _, synthetic_cosine = self._cosine_distance(
            high_target, synthetic_output
        )
        synthetic_cosine = tf.clip_by_value(
            synthetic_cosine, -0.999999, 0.999999
        )
        synthetic_sam_degrees = tf.reduce_mean(
            tf.acos(synthetic_cosine)
            * tf.cast(
                180.0 / 3.141592653589793, synthetic_cosine.dtype
            )
        )
        if self.synthetic_self_similarity_weight > 0:
            synthetic_self_similarity, _ = (
                self.self_similarity_loss_fn.per_sample(
                    high_target, synthetic_output
                )
            )
            synthetic_self_similarity_loss = tf.reduce_mean(
                synthetic_self_similarity
            )
        else:
            synthetic_self_similarity_loss = tf.cast(
                0.0, synthetic_pair_loss.dtype
            )
        return {
            "synthetic_pair_loss": synthetic_pair_loss,
            "synthetic_classification_loss": synthetic_classification_loss,
            "high_quality_identity_loss": high_quality_identity_loss,
            "group_spectral_loss": group_spectral_loss,
            "group_feature_loss": group_feature_loss,
            "synthetic_sam_degrees": synthetic_sam_degrees,
            "synthetic_self_similarity_loss": (
                synthetic_self_similarity_loss
            ),
        }

    def _calculate_unpaired_dual_losses(
        self,
        raw_spectra,
        target_spectra,
        labels,
        target_features,
        target_probabilities,
        target_component_a,
        target_component_b,
        group_ids,
        reference_indices,
        augmentation_index,
        training,
    ):
        values = super()._calculate_dual_losses(
            raw_spectra,
            target_spectra,
            labels,
            target_features,
            target_probabilities,
            target_component_a,
            target_component_b,
            training,
        )
        unpaired = self._unpaired_values(
            values,
            group_ids,
            reference_indices,
            augmentation_index,
            training,
        )
        spectral_auxiliary = (
            self.synthetic_pair_weight * unpaired["synthetic_pair_loss"]
            + self.high_quality_identity_weight
            * unpaired["high_quality_identity_loss"]
            + self.group_spectral_weight
            * unpaired["group_spectral_loss"]
            + self.synthetic_self_similarity_weight
            * unpaired["synthetic_self_similarity_loss"]
        )
        classification_auxiliary = (
            self.synthetic_classification_weight
            * unpaired["synthetic_classification_loss"]
            + self.group_feature_weight * unpaired["group_feature_loss"]
        )
        values.update(unpaired)
        values["spectral_objective"] += spectral_auxiliary
        values["classification_objective"] += classification_auxiliary
        values["loss"] += spectral_auxiliary + classification_auxiliary
        return values

    def _step(self, data, training):
        raw_spectra, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        (
            target_spectra,
            labels,
            target_features,
            target_probabilities,
            target_component_a,
            target_component_b,
        ) = self._unpack_dual_targets(targets)
        group_ids, reference_indices, augmentation_index = (
            self._required_unpaired_targets(targets)
        )

        gradient_cosine = tf.constant(0.0, dtype=tf.float32)
        if training:
            variables = self.denoiser.trainable_variables
            with tf.GradientTape(persistent=self.use_pcgrad) as tape:
                values = self._calculate_unpaired_dual_losses(
                    raw_spectra,
                    target_spectra,
                    labels,
                    target_features,
                    target_probabilities,
                    target_component_a,
                    target_component_b,
                    group_ids,
                    reference_indices,
                    augmentation_index,
                    training=True,
                )
            if self.use_pcgrad:
                spectral = self._dense_gradients(
                    tape.gradient(values["spectral_objective"], variables),
                    variables,
                )
                classification = self._dense_gradients(
                    tape.gradient(
                        values["classification_objective"], variables
                    ),
                    variables,
                )
                unmixing = self._dense_gradients(
                    tape.gradient(values["unmixing_loss"], variables),
                    variables,
                )
                del tape
                classification, unmixing, gradient_cosine = self._pcgrad_pair(
                    classification, unmixing
                )
                gradients = [
                    first + second + third
                    for first, second, third in zip(
                        spectral, classification, unmixing
                    )
                ]
            else:
                gradients = tape.gradient(values["loss"], variables)
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            ]
            self.optimizer.apply_gradients(pairs)
        else:
            values = self._calculate_unpaired_dual_losses(
                raw_spectra,
                target_spectra,
                labels,
                target_features,
                target_probabilities,
                target_component_a,
                target_component_b,
                group_ids,
                reference_indices,
                augmentation_index,
                training=False,
            )
        values["classification_unmixing_gradient_cosine"] = gradient_cosine

        for name, tracker in self.trackers.items():
            tracker.update_state(values[name])
        self.spectral_mae.update_state(target_spectra, values["denoised"])
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)


class DualGuidedRealDomainAdaptationModel(RealDomainAdaptationModel):
    """Adapt a denoiser through independent frozen classifier and unmixer.

    Real plastic mixtures do not provide sample-aligned endpoint or abundance
    truth.  Consequently this model never presents proxy targets as measured
    ground truth.  It preserves the frozen unmixer response on the original
    denoised spectrum (PIT distillation), encourages the two endpoints to match
    the two known pure-spectrum class centroids, and enforces that their sum
    reconstructs the spectrum passed to the unmixer.

    Classification and unmixing remain separate inference models.  Their
    gradients meet only in the small residual denoising head.  Optional PCGrad
    removes the conflicting component of those two task gradients before the
    common spectral-fidelity gradient is added.
    """

    def __init__(
        self,
        denoiser,
        teacher,
        unmixer,
        reference_centroids,
        unmixing_distillation_weight=0.002,
        endpoint_classification_weight=0.001,
        mixture_consistency_guidance_weight=0.05,
        endpoint_temperature=0.10,
        use_pcgrad=True,
        **kwargs,
    ):
        super().__init__(denoiser=denoiser, teacher=teacher, **kwargs)
        self.unmixer = unmixer
        self.unmixer.trainable = False
        centroids = tf.convert_to_tensor(reference_centroids, dtype=tf.float32)
        if centroids.shape.rank == 3 and centroids.shape[-1] == 1:
            centroids = centroids[..., 0]
        if centroids.shape.rank != 2:
            raise ValueError("reference_centroids must have shape (classes, bands)")
        self.reference_centroids = tf.math.l2_normalize(centroids, axis=-1)
        self.unmixing_distillation_weight = float(
            unmixing_distillation_weight
        )
        self.endpoint_classification_weight = float(
            endpoint_classification_weight
        )
        self.mixture_consistency_guidance_weight = float(
            mixture_consistency_guidance_weight
        )
        self.endpoint_temperature = float(endpoint_temperature)
        self.use_pcgrad = bool(use_pcgrad)
        if min(
            self.unmixing_distillation_weight,
            self.endpoint_classification_weight,
            self.mixture_consistency_guidance_weight,
        ) < 0:
            raise ValueError("dual-guidance loss weights must be nonnegative")
        if self.endpoint_temperature <= 0:
            raise ValueError("endpoint_temperature must be positive")

        for name in (
            "unmixing_loss",
            "unmixing_distillation_loss",
            "endpoint_classification_loss",
            "mixture_consistency_guidance_loss",
            "guided_unmix_pair_accuracy",
            "target_unmix_pair_accuracy",
            "guided_endpoint_sam_degrees",
            "target_endpoint_sam_degrees",
            "classification_unmixing_gradient_cosine",
        ):
            self.trackers[name] = tf.keras.metrics.Mean(name=name)

    @staticmethod
    def _unpack_dual_targets(targets):
        base = RealDomainAdaptationModel._unpack_targets(targets)
        required = ("target_component_a", "target_component_b")
        missing = [key for key in required if key not in targets]
        if missing:
            raise ValueError("missing target fields: " + ", ".join(missing))
        return base + tuple(targets[key] for key in required)

    @staticmethod
    def _per_sample_cosine_distance(reference, prediction):
        reference = tf.reshape(reference, (tf.shape(reference)[0], -1))
        prediction = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
        reference = tf.math.l2_normalize(reference, axis=-1)
        prediction = tf.math.l2_normalize(prediction, axis=-1)
        return 1.0 - tf.clip_by_value(
            tf.reduce_sum(reference * prediction, axis=-1), -1.0, 1.0
        )

    def _centroid_logits(self, component):
        component = tf.reshape(component, (tf.shape(component)[0], -1))
        component = tf.math.l2_normalize(component, axis=-1)
        return tf.matmul(
            component, self.reference_centroids, transpose_b=True
        ) / tf.cast(self.endpoint_temperature, component.dtype)

    @staticmethod
    def _true_pair_indices(labels):
        labels = tf.cast(labels, tf.float32)
        positive_count = tf.reduce_sum(
            tf.cast(labels > 0.5, tf.int32), axis=-1
        )
        assertion = tf.debugging.assert_equal(
            positive_count,
            tf.fill(tf.shape(positive_count), 2),
            message="dual-guided plastic training requires exactly two labels",
        )
        with tf.control_dependencies([assertion]):
            return tf.math.top_k(labels, k=2).indices

    @staticmethod
    def _pair_accuracy(logits_a, logits_b, true_indices):
        category_count = tf.shape(logits_a)[1]
        scores = logits_a[:, :, None] + logits_b[:, None, :]
        diagonal = tf.eye(
            category_count, batch_shape=[tf.shape(scores)[0]],
            dtype=scores.dtype
        )
        scores = scores - diagonal * tf.cast(1e6, scores.dtype)
        flattened = tf.reshape(scores, (tf.shape(scores)[0], -1))
        ordered = tf.argmax(flattened, axis=-1, output_type=tf.int32)
        predicted = tf.stack(
            (ordered // category_count, ordered % category_count), axis=-1
        )
        predicted = tf.sort(predicted, axis=-1)
        truth = tf.sort(tf.cast(true_indices, tf.int32), axis=-1)
        return tf.reduce_mean(
            tf.cast(tf.reduce_all(tf.equal(predicted, truth), axis=-1), tf.float32)
        )

    def _centroid_losses_and_metrics(
        self, component_a, component_b, labels
    ):
        logits_a = self._centroid_logits(component_a)
        logits_b = self._centroid_logits(component_b)
        indices = self._true_pair_indices(labels)
        first, second = indices[:, 0], indices[:, 1]
        direct = 0.5 * (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=first, logits=logits_a
            )
            + tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=second, logits=logits_b
            )
        )
        swapped = 0.5 * (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=second, logits=logits_a
            )
            + tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=first, logits=logits_b
            )
        )

        cosine_a = logits_a * tf.cast(
            self.endpoint_temperature, logits_a.dtype
        )
        cosine_b = logits_b * tf.cast(
            self.endpoint_temperature, logits_b.dtype
        )
        rows = tf.range(tf.shape(indices)[0], dtype=tf.int32)
        a_first = tf.gather_nd(cosine_a, tf.stack((rows, first), axis=1))
        a_second = tf.gather_nd(cosine_a, tf.stack((rows, second), axis=1))
        b_first = tf.gather_nd(cosine_b, tf.stack((rows, first), axis=1))
        b_second = tf.gather_nd(cosine_b, tf.stack((rows, second), axis=1))
        direct_cosine = 0.5 * (a_first + b_second)
        swapped_cosine = 0.5 * (a_second + b_first)
        assigned_cosine = tf.maximum(direct_cosine, swapped_cosine)
        safe_cosine = tf.clip_by_value(assigned_cosine, -0.999999, 0.999999)
        sam_degrees = tf.reduce_mean(
            tf.acos(safe_cosine)
            * tf.cast(180.0 / 3.141592653589793, safe_cosine.dtype)
        )
        return {
            "endpoint_classification_loss": tf.reduce_mean(
                tf.minimum(direct, swapped)
            ),
            "pair_accuracy": self._pair_accuracy(logits_a, logits_b, indices),
            "endpoint_sam_degrees": sam_degrees,
        }

    def _unmixing_values(
        self,
        denoised,
        labels,
        target_component_a,
        target_component_b,
    ):
        guided_component_a, guided_component_b = self.unmixer(
            denoised, training=False
        )
        target_component_a = tf.stop_gradient(target_component_a)
        target_component_b = tf.stop_gradient(target_component_b)
        direct = 0.5 * (
            self._per_sample_cosine_distance(
                target_component_a, guided_component_a
            )
            + self._per_sample_cosine_distance(
                target_component_b, guided_component_b
            )
        )
        swapped = 0.5 * (
            self._per_sample_cosine_distance(
                target_component_a, guided_component_b
            )
            + self._per_sample_cosine_distance(
                target_component_b, guided_component_a
            )
        )
        distillation_loss = tf.reduce_mean(tf.minimum(direct, swapped))
        mixture = guided_component_a + guided_component_b
        mixture_loss = tf.reduce_mean(tf.square(mixture - denoised))
        guided_proxy = self._centroid_losses_and_metrics(
            guided_component_a, guided_component_b, labels
        )
        target_proxy = self._centroid_losses_and_metrics(
            target_component_a, target_component_b, labels
        )
        unmixing_loss = (
            self.unmixing_distillation_weight * distillation_loss
            + self.endpoint_classification_weight
            * guided_proxy["endpoint_classification_loss"]
            + self.mixture_consistency_guidance_weight * mixture_loss
        )
        return {
            "unmixing_loss": unmixing_loss,
            "unmixing_distillation_loss": distillation_loss,
            "endpoint_classification_loss": guided_proxy[
                "endpoint_classification_loss"
            ],
            "mixture_consistency_guidance_loss": mixture_loss,
            "guided_unmix_pair_accuracy": guided_proxy["pair_accuracy"],
            "target_unmix_pair_accuracy": target_proxy["pair_accuracy"],
            "guided_endpoint_sam_degrees": guided_proxy[
                "endpoint_sam_degrees"
            ],
            "target_endpoint_sam_degrees": target_proxy[
                "endpoint_sam_degrees"
            ],
        }

    def _calculate_dual_losses(
        self,
        raw_spectra,
        target_spectra,
        labels,
        target_features,
        target_probabilities,
        target_component_a,
        target_component_b,
        training,
    ):
        values = super()._calculate_losses(
            raw_spectra,
            target_spectra,
            labels,
            target_features,
            target_probabilities,
            training,
        )
        unmixing = self._unmixing_values(
            values["denoised"], labels, target_component_a, target_component_b
        )
        regularization = tf.cast(0.0, values["loss"].dtype)
        if self.denoiser.losses:
            regularization = tf.add_n(self.denoiser.losses)
        spectral_objective = (
            self.reconstruction_weight * values["reconstruction_loss"]
            + self.sam_weight * values["sam_loss"]
            + self.correction_weight * values["correction_loss"]
            + self.self_similarity_weight
            * values["self_similarity_loss"]
            + regularization
        )
        classification_objective = (
            self.classification_weight * values["classification_loss"]
            + self.distillation_weight * values["distillation_loss"]
            + self.feature_weight * values["feature_loss"]
            + self.safe_margin_weight * values["safe_margin_loss"]
        )
        values.update(unmixing)
        values["spectral_objective"] = spectral_objective
        values["classification_objective"] = classification_objective
        values["loss"] = (
            spectral_objective
            + classification_objective
            + unmixing["unmixing_loss"]
        )
        return values

    @staticmethod
    def _dense_gradients(gradients, variables):
        dense = []
        for gradient, variable in zip(gradients, variables):
            if gradient is None:
                gradient = tf.zeros_like(variable)
            elif isinstance(gradient, tf.IndexedSlices):
                gradient = tf.convert_to_tensor(gradient)
            dense.append(gradient)
        return dense

    @staticmethod
    def _gradient_dot(first, second):
        return tf.add_n(
            [tf.reduce_sum(a * b) for a, b in zip(first, second)]
        )

    @classmethod
    def _pcgrad_pair(cls, first, second):
        dot = cls._gradient_dot(first, second)
        epsilon = tf.cast(tf.keras.backend.epsilon(), dot.dtype)
        first_norm = cls._gradient_dot(first, first)
        second_norm = cls._gradient_dot(second, second)
        negative_dot = tf.minimum(dot, tf.cast(0.0, dot.dtype))
        projected_first = [
            a - negative_dot / (second_norm + epsilon) * b
            for a, b in zip(first, second)
        ]
        projected_second = [
            b - negative_dot / (first_norm + epsilon) * a
            for a, b in zip(first, second)
        ]
        cosine = dot / (tf.sqrt(first_norm * second_norm) + epsilon)
        return projected_first, projected_second, cosine

    def _step(self, data, training):
        raw_spectra, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        (
            target_spectra,
            labels,
            target_features,
            target_probabilities,
            target_component_a,
            target_component_b,
        ) = self._unpack_dual_targets(targets)

        gradient_cosine = tf.constant(0.0, dtype=tf.float32)
        if training:
            variables = self.denoiser.trainable_variables
            with tf.GradientTape(persistent=self.use_pcgrad) as tape:
                values = self._calculate_dual_losses(
                    raw_spectra,
                    target_spectra,
                    labels,
                    target_features,
                    target_probabilities,
                    target_component_a,
                    target_component_b,
                    training=True,
                )
            if self.use_pcgrad:
                spectral = self._dense_gradients(
                    tape.gradient(values["spectral_objective"], variables),
                    variables,
                )
                classification = self._dense_gradients(
                    tape.gradient(
                        values["classification_objective"], variables
                    ),
                    variables,
                )
                unmixing = self._dense_gradients(
                    tape.gradient(values["unmixing_loss"], variables), variables
                )
                del tape
                classification, unmixing, gradient_cosine = self._pcgrad_pair(
                    classification, unmixing
                )
                gradients = [
                    a + b + c
                    for a, b, c in zip(spectral, classification, unmixing)
                ]
            else:
                gradients = tape.gradient(values["loss"], variables)
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            ]
            self.optimizer.apply_gradients(pairs)
        else:
            values = self._calculate_dual_losses(
                raw_spectra,
                target_spectra,
                labels,
                target_features,
                target_probabilities,
                target_component_a,
                target_component_b,
                training=False,
            )
        values["classification_unmixing_gradient_cosine"] = gradient_cosine

        for name, tracker in self.trackers.items():
            tracker.update_state(values[name])
        self.spectral_mae.update_state(target_spectra, values["denoised"])
        return {metric.name: metric.result() for metric in self.metrics}
