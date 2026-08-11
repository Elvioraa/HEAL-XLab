# DS-V1.1 Residual Identity Correction

DS-V1.1 is a single-variable follow-up to DS-V1. It changes only the yaw
residual identity contract. It does not add multi-scale features, quality
weighting, mixed proposals, remote proposal rescue, or architectural changes.

## Residual Contract

Legacy DS-V1 uses:

```text
[sin(dyaw), cos(dyaw)]
```

Its identity target is `[0, 1]`, while the zero-initialized shared geometry
refiner initially predicts `[0, 0]`.

DS-V1.1 uses:

```text
[sin(dyaw), cos(dyaw) - 1]
```

and decodes with:

```text
dyaw = atan2(yaw_sin, yaw_cos_centered + 1)
```

The eight-dimensional identity residual is therefore exactly all zeros and is
aligned with the refiner's zero-initialized output layer.

## Experimental Hypothesis

Removing the constant cosine target should let translation, size, and yaw
geometry channels receive meaningful learning signal instead of spending the
initial optimization budget learning the identity cosine offset. DS-V1.1 must
be compared directly with DS-V1 under identical detector, proposal, loss,
optimizer, scheduler, epoch, batch, accumulation, and AMP settings.

## Compatibility

`yaw_mode: sin_cos` remains the legacy DS-V1 contract.
`yaw_mode: sin_cos_centered` selects DS-V1.1. The two modes have identical
state-dict keys and tensor shapes but incompatible residual semantics. A
DS-V1 checkpoint must not be loaded under a DS-V1.1 configuration.

DS-V1.1 requires a new Stage1 run followed by new m2, m3, and m4 Stage2 runs
seeded from that Stage1 checkpoint. The normal merge ownership remains:
shared object-space modules from Stage1 and modality adapters from the matching
Stage2 checkpoints.
