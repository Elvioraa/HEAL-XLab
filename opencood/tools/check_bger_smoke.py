"""Smoke test for the BGER (Box-Guided Evidence Reactivation) line.

CPU-only, no dataset / checkpoint required. Checks:

1. yaml sanity for all HEAL_XLab_v4_BGER configs.
2. bger.enabled=false  =>  bit-exact equivalence with HeterPyramidCollab.
3. oracle-mode forward: shapes, finiteness, comm accounting, empty-collab.
4. single_decode-mode forward: box decoding path end to end.
5. BGERRefine zero-init identity.
6. BGERBoxPrior rendering & collaborator->ego box projection geometry.
7. freeze_base: gradients reach only bger_refine; frozen BN stays eval
   after model.train().
"""

import os
import sys
import types

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if "opencood.utils.box_overlaps" not in sys.modules:
    box_overlaps_stub = types.ModuleType("opencood.utils.box_overlaps")
    box_overlaps_stub.bbox_overlaps = lambda *args, **kwargs: None
    sys.modules["opencood.utils.box_overlaps"] = box_overlaps_stub
if "icecream" not in sys.modules:
    icecream_stub = types.ModuleType("icecream")
    icecream_stub.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = icecream_stub
try:
    import open3d  # noqa: F401
    import cv2  # noqa: F401
except ImportError:
    # single_decode lazily imports VoxelPostprocessor, whose module chain
    # pulls in opencood.visualization.vis_utils (open3d / cv2). Stub it so
    # the smoke test stays CPU/dependency-light.
    vis_utils_stub = types.ModuleType("opencood.visualization.vis_utils")
    sys.modules["opencood.visualization.vis_utils"] = vis_utils_stub

from opencood.hypes_yaml import yaml_utils
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_bger import HeterPyramidCollabBger
from opencood.models.sub_modules.bger_box_prior import BGERBoxPrior
from opencood.models.sub_modules.bger_refine import BGERRefine


CHANNELS = 32
FEAT_H = 8
FEAT_W = 8
# 2 m cells: x in [-8, 8] over 8 cols, y in [-8, 8] over 8 rows.
LIDAR_RANGE = [-8.0, -8.0, -3.0, 8.0, 8.0, 1.0]

YAML_STAGE_A = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_BGER",
    "stage_a", "m1_bger_oracle.yaml")
YAML_STAGE_B = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_BGER",
    "stage_b", "m1_ego_heter_bger_single.yaml")
YAML_INFER = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_BGER",
    "final_infer", "m1_ego_m2m3m4_bger.yaml")
YAML_BOXMERGE = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_BGER",
    "final_infer", "m1_ego_m2m3m4_boxmerge.yaml")


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, data_dict, modality_name):
        return data_dict[f"inputs_{modality_name}"]["feature"] * self.scale


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch_dict):
        return {"spatial_features_2d": batch_dict["spatial_features"] * self.scale}


class _DummyAligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, feature):
        return feature * self.scale


class _DummyPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.align_corners = False
        self.scale = nn.Parameter(torch.ones(()))

    def forward_single(self, feature):
        feature = feature * self.scale
        return feature, [feature[:, :1]]

    def forward_collab(self, feature, record_len, affine_matrix,
                       agent_modality_list=None, cam_crop_info=None):
        rows = []
        start = 0
        for length in record_len.detach().cpu().view(-1).tolist():
            rows.append(feature[start].clone())
            start += int(length)
        return torch.stack(rows, dim=0) * self.scale, [feature[:, :1]]


def _install_base_stub_modules(model, modality_names):
    model.modality_name_list = list(modality_names)
    model.sensor_type_dict = {name: "lidar" for name in modality_names}
    model.cam_crop_info = {}
    model.H = float(LIDAR_RANGE[4] - LIDAR_RANGE[1])
    model.W = float(LIDAR_RANGE[3] - LIDAR_RANGE[0])
    model.fake_voxel_size = 1
    model.compress = False
    model.shrink_flag = False
    model.cav_range = LIDAR_RANGE
    model.args = {"dir_args": {"dir_offset": 0.7853, "num_bins": 2}}
    for name in modality_names:
        setattr(model, f"encoder_{name}", _DummyEncoder())
        setattr(model, f"backbone_{name}", _DummyBackbone())
        setattr(model, f"aligner_{name}", _DummyAligner())
        setattr(model, f"depth_supervision_{name}", False)
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(CHANNELS, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(CHANNELS, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(CHANNELS, 4, kernel_size=1)


def _build_plain_model(modality_names):
    model = HeterPyramidCollab.__new__(HeterPyramidCollab)
    nn.Module.__init__(model)
    _install_base_stub_modules(model, modality_names)
    return model


def _build_bger_model(bger_cfg, modality_names=("m1",)):
    model = HeterPyramidCollabBger.__new__(HeterPyramidCollabBger)
    nn.Module.__init__(model)
    _install_base_stub_modules(model, modality_names)
    model.supervise_single = False
    model.bger_cfg = HeterPyramidCollabBger._normalize_bger_cfg(bger_cfg)
    model.bger_enabled = bool(model.bger_cfg["enabled"])
    model.bger_freeze_base = bool(model.bger_cfg["freeze_base"])
    if model.bger_enabled:
        prior_cfg = dict(model.bger_cfg["prior"])
        prior_cfg["lidar_range"] = LIDAR_RANGE
        model.bger_box_prior = BGERBoxPrior(prior_cfg)
        refine_cfg = dict(model.bger_cfg["refine"])
        refine_cfg["in_channels"] = CHANNELS
        refine_cfg["prior_channels"] = model.bger_box_prior.num_channels
        model.bger_refine = BGERRefine(refine_cfg)
        if model.bger_freeze_base:
            model._freeze_base_parameters()
    return model


def _pairwise_with_offset(batch_size, max_cav, dx=0.0, dy=0.0):
    """Identity everywhere except collaborator(local 1) -> ego at [b,1,0]."""
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(
        batch_size, max_cav, max_cav, 1, 1
    )
    if max_cav > 1:
        for b in range(batch_size):
            pairwise[b, 1, 0, 0, 3] = dx
            pairwise[b, 1, 0, 1, 3] = dy
            pairwise[b, 0, 1, 0, 3] = -dx
            pairwise[b, 0, 1, 1, 3] = -dy
    return pairwise


def _dummy_data(record_len, max_num=10, with_oracle=True, with_anchor=False,
                collab_box_xy=(0.0, 0.0), offset=(0.0, 0.0)):
    record_len = torch.as_tensor(record_len)
    total_agents = int(record_len.sum())
    max_cav = int(record_len.max())
    data = {
        "agent_modality_list": ["m1"] * total_agents,
        "record_len": record_len,
        "pairwise_t_matrix": _pairwise_with_offset(
            len(record_len), max_cav, dx=offset[0], dy=offset[1]
        ),
        "inputs_m1": {
            "feature": torch.randn(total_agents, CHANNELS, FEAT_H, FEAT_W,
                                   requires_grad=True),
        },
    }
    if with_oracle:
        centers = torch.zeros(total_agents, max_num, 7)
        mask = torch.zeros(total_agents, max_num)
        flat = 0
        for length in record_len.tolist():
            for local in range(int(length)):
                if local == 0:
                    flat += 1
                    continue
                centers[flat, 0] = torch.tensor(
                    [collab_box_xy[0], collab_box_xy[1], -1.0,
                     1.56, 1.6, 3.9, 0.0]
                )
                mask[flat, 0] = 1.0
                flat += 1
        data["object_bbx_center_single"] = centers
        data["object_bbx_mask_single"] = mask
    if with_anchor:
        xs = torch.linspace(LIDAR_RANGE[0] + 1.0, LIDAR_RANGE[3] - 1.0, FEAT_W)
        ys = torch.linspace(LIDAR_RANGE[1] + 1.0, LIDAR_RANGE[4] - 1.0, FEAT_H)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        anchor = torch.zeros(FEAT_H, FEAT_W, 2, 7)
        anchor[..., 0] = gx.unsqueeze(-1)
        anchor[..., 1] = gy.unsqueeze(-1)
        anchor[..., 2] = -1.0
        anchor[..., 3] = 1.56
        anchor[..., 4] = 1.6
        anchor[..., 5] = 3.9
        anchor[..., 6] = torch.tensor([0.0, 1.5707]).view(1, 1, 2)
        data["anchor_box"] = anchor
    return data


def _check_yaml_configs():
    stage_a = yaml_utils.load_yaml(YAML_STAGE_A)
    assert stage_a["name"] == "BGER_v1/stage_a/m1_bger_oracle"
    assert stage_a["model"]["core_method"] == "heter_pyramid_collab_bger"
    assert stage_a["fusion"]["core_method"] == "intermediateheter"
    cfg = stage_a["model"]["args"]["bger"]
    assert cfg["enabled"] is True
    assert cfg["box_source"] == "oracle"
    assert cfg["mode"] == "refine"
    assert cfg["freeze_base"] is True
    assert stage_a["loss"]["core_method"] == "point_pillar_pyramid_loss"

    stage_b = yaml_utils.load_yaml(YAML_STAGE_B)
    assert stage_b["name"] == "BGER_v1/stage_b/m1_ego_heter_bger_single"
    assert stage_b["model"]["core_method"] == "heter_pyramid_collab_bger"
    cfg = stage_b["model"]["args"]["bger"]
    assert cfg["enabled"] is True
    assert cfg["box_source"] == "single_decode"
    assert cfg["mode"] == "refine"
    for modality in ("m1", "m2", "m3", "m4"):
        assert modality in stage_b["model"]["args"]
        assert stage_b["heter"]["mapping_dict"][modality] == modality
    assert stage_b["heter"]["ego_modality"] == "m1"

    infer = yaml_utils.load_yaml(YAML_INFER)
    assert infer["name"] == "BGER_v1/final_infer/m1_ego_m2m3m4_bger"
    assert infer["model"]["args"]["bger"]["mode"] == "refine"

    boxmerge = yaml_utils.load_yaml(YAML_BOXMERGE)
    assert boxmerge["name"] == "BGER_v1/final_infer/m1_ego_m2m3m4_boxmerge"
    assert boxmerge["model"]["args"]["bger"]["mode"] == "box_merge_only"
    print("BGER yaml configs OK")


def _check_disabled_equivalence():
    torch.manual_seed(11)
    plain = _build_plain_model(["m1"])
    bger = _build_bger_model({"enabled": False}, modality_names=["m1"])
    bger.load_state_dict(plain.state_dict(), strict=False)
    assert not bger.bger_enabled
    assert not hasattr(bger, "bger_refine")

    data = _dummy_data([2, 2], with_oracle=False)
    plain.eval()
    bger.eval()
    with torch.no_grad():
        out_plain = plain(data)
        out_bger = bger(data)
    for key in ("cls_preds", "reg_preds", "dir_preds"):
        assert torch.equal(out_plain[key], out_bger[key]), key
    assert "bger" not in out_bger
    print("BGER disabled == official HeterPyramidCollab OK")


def _check_refine_identity():
    torch.manual_seed(5)
    refine = BGERRefine({
        "in_channels": CHANNELS,
        "prior_channels": 2,
        "hidden_dim": 16,
        "num_layers": 2,
        "norm": "bn",
        "gate_init": 1.0,
    })
    refine.eval()
    feature = torch.randn(2, CHANNELS, FEAT_H, FEAT_W)
    prior = torch.rand(2, 2, FEAT_H, FEAT_W)
    refined, delta = refine(feature, prior)
    assert torch.allclose(refined, feature)
    assert torch.allclose(delta, torch.zeros_like(delta))
    print("BGER refine zero-init identity OK")


def _check_prior_rendering():
    prior_module = BGERBoxPrior({
        "lidar_range": LIDAR_RANGE,
        "gaussian": True,
        "box_mask": True,
        "yaw": True,
        "sigma_scale": 0.25,
        "min_sigma": 1.0,
    })
    assert prior_module.num_channels == 4
    # 4 m x 2 m box at metric (3, 3): exactly on the cell center of
    # (row 5, col 5) for 2 m cells spanning [-8, 8].
    boxes = torch.tensor([[3.0, 3.0, -1.0, 1.5, 2.0, 4.0, 0.0]])
    scores = torch.tensor([0.8])
    prior = prior_module([boxes], [scores], (FEAT_H, FEAT_W), boxes.device,
                         torch.float32)
    assert prior.shape == (1, 4, FEAT_H, FEAT_W)
    gaussian = prior[0, 0]
    peak = (gaussian == gaussian.max()).nonzero()[0]
    assert peak.tolist() == [5, 5], peak.tolist()
    assert abs(float(gaussian.max()) - 0.8) < 1e-4
    mask = prior[0, 1]
    assert float(mask[5, 5]) == 0.8
    assert float(mask.sum()) > 0
    assert abs(float(prior[0, 2, 5, 5]) - 1.0) < 1e-5  # cos(0)
    assert abs(float(prior[0, 3, 5, 5])) < 1e-5        # sin(0)

    empty = prior_module([boxes.new_zeros((0, 7))], [scores.new_zeros((0,))],
                         (FEAT_H, FEAT_W), boxes.device, torch.float32)
    assert float(empty.abs().sum()) == 0.0
    print("BGER prior rendering OK")


def _check_oracle_forward():
    torch.manual_seed(23)
    model = _build_bger_model({
        "enabled": True,
        "box_source": "oracle",
        "freeze_base": True,
    })
    model.eval()
    # collaborator box at its own origin; collaborator->ego offset (4, 2)
    # must land in ego frame at metric (4, 2) -> grid (row 5, col 6).
    data = _dummy_data([2], with_oracle=True,
                       collab_box_xy=(0.0, 0.0), offset=(4.0, 2.0))
    with torch.no_grad():
        output = model(data)
    assert output["cls_preds"].shape == (1, 2, FEAT_H, FEAT_W)
    assert output["reg_preds"].shape == (1, 14, FEAT_H, FEAT_W)
    assert output["dir_preds"].shape == (1, 4, FEAT_H, FEAT_W)
    for key in ("cls_preds", "reg_preds", "dir_preds"):
        assert torch.isfinite(output[key]).all()
    debug = output["bger"]
    assert debug["box_source"] == "oracle"
    assert debug["num_collab_boxes"] == [1]
    box_ego = debug["collab_boxes_ego_frame"][0]
    assert abs(float(box_ego[0, 0]) - 4.0) < 1e-3
    assert abs(float(box_ego[0, 1]) - 2.0) < 1e-3
    assert debug["comm_bytes_boxes"] == 1 * 8 * 4
    assert debug["comm_bytes_feature_equiv"] == \
        1 * CHANNELS * FEAT_H * FEAT_W * 4
    assert 0.0 < debug["comm_ratio"] < 1.0

    # ego-only sample: no collaborators, prior must be empty, forward OK.
    solo = _dummy_data([1], with_oracle=True)
    with torch.no_grad():
        solo_out = model(solo)
    assert solo_out["cls_preds"].shape == (1, 2, FEAT_H, FEAT_W)
    assert solo_out["bger"]["num_collab_boxes"] == [0]
    print("BGER oracle forward OK")


def _check_single_decode_forward():
    torch.manual_seed(31)
    model = _build_bger_model({
        "enabled": True,
        "box_source": "single_decode",
        "freeze_base": True,
        "single_decode": {"score_threshold": 0.05, "nms_thresh": 0.15,
                          "max_boxes": 5},
    })
    model.eval()
    data = _dummy_data([2, 2], with_oracle=False, with_anchor=True)
    with torch.no_grad():
        output = model(data)
    assert output["cls_preds"].shape == (2, 2, FEAT_H, FEAT_W)
    debug = output["bger"]
    assert debug["box_source"] == "single_decode"
    assert len(debug["num_collab_boxes"]) == 2
    for boxes in debug["collab_boxes_ego_frame"]:
        assert boxes.shape[0] <= 5
        assert boxes.shape[1:] == (7,) if boxes.shape[0] > 0 else True
    print("BGER single_decode forward OK")


def _check_gradient_isolation():
    torch.manual_seed(47)
    model = _build_bger_model({
        "enabled": True,
        "box_source": "oracle",
        "freeze_base": True,
    })
    # add a base-side BN to verify the train() override keeps it eval
    model.frozen_bn_probe = nn.BatchNorm2d(3)
    for param in model.frozen_bn_probe.parameters():
        param.requires_grad_(False)
    model.train()
    assert model.frozen_bn_probe.training is False
    refine_bns = [m for m in model.bger_refine.modules()
                  if isinstance(m, nn.BatchNorm2d)]
    assert refine_bns and all(m.training for m in refine_bns)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("bger_refine.") for name in trainable), trainable

    data = _dummy_data([2], with_oracle=True, collab_box_xy=(1.0, 1.0))
    output = model(data)
    loss = output["cls_preds"].mean() + output["reg_preds"].abs().mean()
    loss.backward()
    got_refine_grad = False
    for name, param in model.named_parameters():
        if name.startswith("bger_refine.") and param.grad is not None \
                and float(param.grad.abs().sum()) > 0:
            got_refine_grad = True
        if not name.startswith("bger_refine."):
            assert param.grad is None or float(param.grad.abs().sum()) == 0.0, name
    assert got_refine_grad
    print("BGER gradient isolation OK")


def main():
    _check_yaml_configs()
    _check_disabled_equivalence()
    _check_refine_identity()
    _check_prior_rendering()
    _check_oracle_forward()
    _check_single_decode_forward()
    _check_gradient_isolation()
    print("BGER smoke OK")


if __name__ == "__main__":
    main()
