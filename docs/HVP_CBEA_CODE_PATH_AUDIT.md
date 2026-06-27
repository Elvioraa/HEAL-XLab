# HVP-CBEA Code Path Audit

Base hash before implementation: `fb533c2`

## Model Registry

- Model creation is implemented in `opencood/tools/train_utils.py:create_model`.
- The registry is dynamic: `hypes['model']['core_method']` is imported as `opencood.models.<core_method>`.
- The class is found by lowercasing and removing underscores from the `core_method`.
- `opencood/models/__init__.py` is not the active model registry in this repository.
- Safe v2 integration path is therefore to add `opencood/models/heter_pyramid_collab_hvp_cbea.py` with class `HeterPyramidCollabHvpCbea` and set:
  - `model.core_method: heter_pyramid_collab_hvp_cbea`
  - `model.args.hvp_cbea.enabled: true`

## Official HEAL Model Path

- Current HEAL model file: `opencood/models/heter_pyramid_collab.py`.
- Main class: `HeterPyramidCollab`.
- `__init__` receives `hypes['model']['args']` as `args`.
- Modality encoders/backbones/aligners are built from `args['m1']`, `args['m2']`, etc.
- The existing official model reads `args['lidar_range']`, `args['fusion_backbone']`, `args['in_head']`, `args['anchor_number']`, and `args['dir_args']`.

## Forward Variables

Real forward variables in `HeterPyramidCollab.forward(data_dict)`:

- `agent_modality_list = data_dict['agent_modality_list']`
- `affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)`
- `record_len = data_dict['record_len']`
- Per-modality BEV feature after encoder/backbone/aligner: local variable `feature`.
- Per-agent BEV features assembled before fusion: `heter_feature_2d`.
- Official fusion call:
  - `self.pyramid_backbone.forward_collab(heter_feature_2d, record_len, affine_matrix, agent_modality_list, self.cam_crop_info)`
- Feature immediately before `cls_head`, `reg_head`, `dir_head`: `fused_feature`.
- Output dictionary contains:
  - `pyramid: collab`
  - optional `depth_items_<modality>`
  - `cls_preds`
  - `reg_preds`
  - `dir_preds`
  - `occ_single_list`

## Feature Availability

- Ego BEV feature can be identified as the first agent feature in each scene segment of `heter_feature_2d`, using `record_len`.
- Collaborator BEV features are the remaining features in each scene segment.
- Because this is heterogeneous intermediate fusion, raw camera images are already consumed by official encoders before HVP-CBEA sees BEV tensors. HVP-CBEA does not bypass or access raw camera features.
- The stable injection point for v2 is after official `pyramid_backbone.forward_collab(...)` and optional `shrink_conv`, before the official detection heads.

## GT and Loss Path

- No `batch_dict['gt_boxes']` style field is used by the official model forward.
- Dataset labels are collated into `batch_data['ego']['label_dict']`; the external loss receives this as `target_dict`.
- Training calls in `opencood/tools/train.py`:
  - `ouput_dict = model(batch_data['ego'])`
  - `final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])`
- Current HEAL final-infer yaml uses `loss.core_method: point_pillar_depth_loss`.
- `PointPillarDepthLoss` inherits from `PointPillarLoss`.
- Safe loss integration point is `opencood/loss/point_pillar_loss.py`: add `output_dict['hvp_cbea_loss']` only when present.
- If `hvp_cbea_loss` is absent, official loss remains equivalent.
- HVP-CBEA submodule `compute_loss` methods must return zero if GT format is unavailable or unsafe.

## Integration Decision

- Chosen plan: B.
- Add a new model core method `heter_pyramid_collab_hvp_cbea` instead of modifying official `heter_pyramid_collab`.
- Official v1 HBEC post-process code remains untouched.
- Official HEAL model path remains untouched unless yaml explicitly selects the new core method.

