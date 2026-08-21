# Dual-Space V3 Optional Extensions

The Dual-Space core remains `version: ds_v3`. Diagnostics, V5, and V6 are
independent extensions and are disabled when their parent field is absent or
has `enabled: false`.

## Runtime Contract

- `diagnostics` is observer-only. Training JSONL is written by rank 0 to
  `<model_dir>/diagnostics/quality_target.jsonl` only when both
  `diagnostics.enabled` and `diagnostics.quality_target.enabled` are true.
- `v5_quality_safe` is active only in `mode: stage2_adapt`. It adds no module,
  parameter, buffer, or checkpoint key.
- `v6_residual_safe` is active only in `mode: stage2_adapt` and
  `mode: inference`. It transforms the existing object/context adapter
  residual and adds no state.
- Diagnostic inference bypasses require both diagnostics parent gates and are
  rejected outside `mode: inference`.

With all extension parents disabled, the original V3 adapter call, consensus,
quality loss, parameter trainability, state dictionary, and tensor outputs are
preserved.

## V5 Quality Safety

The optional valid mask reuses the proposal sampler's existing GT assignment:

```text
valid = matched AND finite(target) AND min_target <= target <= max_target
```

No-valid batches return `quality_pred.sum() * 0`, preserving graph/device
connectivity. The ratio cap is:

```text
base_quality = quality_loss_weight * raw_quality_loss
cap = max_quality_to_detection_ratio * detach(detection_objective)
scale = min(1, cap / (detach(base_quality) + eps))
balanced_quality = scale * base_quality
```

Ranking pairs are built only within one scene and one agent. Pairs are selected
in deterministic agent/proposal order and use
`softplus(-sign(target_i-target_j) * (quality_i-quality_j))`.

## V6 Residual Safety

Both real adapters consume ROI tensors `[M,C,Rh,Rw]`; channel dimension `1` is
the explicit feature dimension. For input `x` and existing residual `r`:

```text
ratio = ||r||_C / (||x||_C + eps)
cap_scale = min(1, max_residual_ratio / (ratio + eps))
r_safe = residual_scale * cap_scale * r
output = x + r_safe
```

Object and context branches are independently gated. The merged inference YAML
must preserve the same V6 settings used by every Stage2 modality.

## Experiment Packs

The following complete five-file packs copy the formal DS-V3 settings without
modifying the original configs:

- `DS_V3_DIAG`
- `DS_V5_QUALITY_SAFE`
- `DS_V6_RESIDUAL_SAFE`
- `DS_V5_V6`

Each pack contains `stage1_m1.yaml`, `stage2_m2.yaml`, `stage2_m3.yaml`,
`stage2_m4.yaml`, and `merged_infer.yaml`.

## Merge Ownership

V5 and V6 add no state-dict key, so production `merge_final` ownership remains
unchanged. The read-only checker is:

```bash
python -m opencood.tools.check_dual_space_merge_ownership \
  --m1 <stage1.pth> --m2 <m2.pth> --m3 <m3.pth> --m4 <m4.pth> \
  --merged <merged.pth>
```
