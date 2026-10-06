"""Canonical loss-group definitions for the denoiser ablation study.

The functions in this module intentionally have no TensorFlow dependency so
the experiment contract can be unit-tested before starting expensive runs.
"""


ABLATION_COMPONENTS = {
    "A": {
        "bounded_residual": False,
        "peak_shape": False,
        "recognition": False,
        "multi_quality": False,
        "endpoint_guidance": False,
        "pcgrad": False,
        "training": False,
        "description": "Frozen paper/base denoiser; derived from the matched baseline predictions.",
    },
    "C": {
        "bounded_residual": True,
        "peak_shape": True,
        "recognition": False,
        "multi_quality": False,
        "endpoint_guidance": False,
        "pcgrad": False,
        "training": True,
        "description": "Bounded spectral residual with peak, derivative, SAM and correction constraints.",
    },
    "D": {
        "bounded_residual": True,
        "peak_shape": True,
        "recognition": True,
        "multi_quality": False,
        "endpoint_guidance": False,
        "pcgrad": False,
        "training": True,
        "description": "C plus classification, distillation, feature and margin guidance.",
    },
    "E": {
        "bounded_residual": True,
        "peak_shape": True,
        "recognition": True,
        "multi_quality": True,
        "endpoint_guidance": False,
        "pcgrad": False,
        "training": True,
        "description": "D plus unpaired multi-quality references, corruption and group alignment.",
    },
    "G": {
        "bounded_residual": True,
        "peak_shape": True,
        "recognition": True,
        "multi_quality": True,
        "endpoint_guidance": True,
        "pcgrad": True,
        "training": True,
        "description": "Complete model: E plus endpoint guidance and PCGrad.",
    },
}


RECOGNITION_WEIGHTS = (
    "classification_weight",
    "distillation_weight",
    "feature_weight",
    "safe_margin_weight",
)

MULTI_QUALITY_WEIGHTS = (
    "synthetic_pair_weight",
    "synthetic_classification_weight",
    "high_quality_identity_weight",
    "group_spectral_weight",
    "group_feature_weight",
)

ENDPOINT_WEIGHTS = (
    "unmixing_distillation_weight",
    "endpoint_classification_weight",
    "mixture_consistency_guidance_weight",
)


def apply_ablation_variant(args):
    """Apply a C/D/E/G ablation contract to an argparse namespace.

    All trainable variants keep the frozen endpoint network loaded so that
    validation checkpoint constraints and endpoint-proxy reporting are
    identical. Disabled groups have exactly zero loss weight and therefore do
    not contribute gradients.
    """
    variant = getattr(args, "ablation_variant", None)
    if variant is None:
        return args
    if variant not in {"C", "D", "E", "G"}:
        raise ValueError("Training supports ablation variants C, D, E and G; A requires no training")
    if not getattr(args, "unmixing_weights", None):
        raise ValueError(
            "Ablation variants require --unmixing-weights for identical "
            "checkpoint selection and endpoint-proxy evaluation"
        )

    components = ABLATION_COMPONENTS[variant]
    args.fine_tune_backbone = False
    args.multi_quality_unpaired = components["multi_quality"]
    args.use_pcgrad = components["pcgrad"]
    # The published A/C/D/E/G contract predates the optional structural
    # self-similarity extension.  Keep it disabled so historical ablations
    # remain exactly reproducible; evaluate it in a separate matched run.
    args.self_similarity_weight = 0.0
    args.synthetic_self_similarity_weight = 0.0

    if not components["recognition"]:
        for name in RECOGNITION_WEIGHTS:
            setattr(args, name, 0.0)
        args.hard_example_weight = 1.0
        args.easy_identity_weight = 1.0

    if not components["multi_quality"]:
        for name in MULTI_QUALITY_WEIGHTS:
            setattr(args, name, 0.0)

    if not components["endpoint_guidance"]:
        for name in ENDPOINT_WEIGHTS:
            setattr(args, name, 0.0)

    args.ablation_components = dict(components)
    args.ablation_contract = (
        "All variants use the same frozen base denoiser, recognition teacher, "
        "endpoint teacher, duplicate-group split and validation checkpoint "
        "constraints. A disabled loss group has zero gradient weight."
    )
    return args


def active_weight_groups(args):
    """Return a compact audit record after overrides have been applied."""
    return {
        "recognition": {name: float(getattr(args, name)) for name in RECOGNITION_WEIGHTS},
        "multi_quality": {name: float(getattr(args, name)) for name in MULTI_QUALITY_WEIGHTS},
        "endpoint": {name: float(getattr(args, name)) for name in ENDPOINT_WEIGHTS},
        "peak_shape": {
            "sam_weight": float(args.sam_weight),
            "peak_weight_scale": float(args.peak_weight_scale),
            "sparse_loss_weight": float(args.sparse_loss_weight),
            "derivative_loss_weight": float(args.derivative_loss_weight),
            "self_similarity_weight": float(args.self_similarity_weight),
            "synthetic_self_similarity_weight": float(
                args.synthetic_self_similarity_weight
            ),
        },
        "use_pcgrad": bool(args.use_pcgrad),
    }
