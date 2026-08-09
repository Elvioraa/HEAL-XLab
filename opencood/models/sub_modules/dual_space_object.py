"""Full Dual-Space HEAL object-space modules and runtime plumbing.

DS-V1 keeps HEAL's Common BEV detector unchanged and adds a second shared
space for proposal geometry.  Modality-specific residual adapters feed one
shared object encoder and one shared geometry refiner.  Multi-agent agreement
happens only after decoding each agent into the common 8D residual space.
Optional multi-scale representation, scalar quality consensus, mixed detector
proposals, and inference-only remote proposal rescue are explicit extensions;
with those switches disabled this module preserves the DS-V1 graph and state.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.dual_space_box_coder import (
    aligned_rotated_bev_iou_hwl,
    boxes_hwl_to_corners_3d,
    corners_3d_to_boxes_hwl,
    decode_box_residual,
    encode_box_residual,
)
from opencood.models.sub_modules.dual_space_config import (
    dual_space_feature_flags,
    validate_dual_space_config,
)
from opencood.models.sub_modules.dual_space_object_roi import (
    ChunkedRotatedBEVROISampler,
    DualSpaceBEVGeometry,
)
from opencood.models.sub_modules.dual_space_proposal_sampler import (
    DualSpaceDetectorProposalDecoder,
    DualSpaceTrainingProposalSampler,
)
from opencood.models.sub_modules.dual_space_remote_proposal_rescue import (
    rescue_remote_proposals,
)


SHARED_DUAL_SPACE_PREFIXES = (
    "dual_space_shared_object_encoder.",
    "dual_space_shared_geometry_encoder.",
    "dual_space_shared_object_refiner.",
    "dual_space_shared_context_encoder.",
    "dual_space_shared_multiscale_fusion.",
    "dual_space_shared_scale_gate.",
    "dual_space_shared_quality_head.",
)

CORE_SHARED_DUAL_SPACE_PREFIXES = SHARED_DUAL_SPACE_PREFIXES[:3]
MULTISCALE_SHARED_DUAL_SPACE_PREFIXES = SHARED_DUAL_SPACE_PREFIXES[3:6]
QUALITY_SHARED_DUAL_SPACE_PREFIXES = SHARED_DUAL_SPACE_PREFIXES[6:]
LOW_QUALITY_REPORT_THRESHOLD = 0.25


class ResidualObjectAdapter(nn.Module):
    """Lightweight modality adapter initialized as an exact identity."""

    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.delta = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, roi_features):
        """Return identity plus a learned residual for ``[M,C,Rh,Rw]``."""
        return roi_features + self.delta(roi_features)


class SharedObjectEncoder(nn.Module):
    """Map adapted Common-BEV ROIs into one shared object representation."""

    def __init__(self, in_channels, hidden_channels, pooled_size, embedding_dim):
        super().__init__()
        groups = _group_count(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_size * pooled_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, roi_features):
        """Encode ``[M,C,Rh,Rw]`` as ``[M,embedding_dim]``."""
        return self.projection(self.features(roi_features))


class SharedGeometryEncoder(nn.Module):
    """Encode normalized proposal geometry from 8D to a shared 32D code."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, raw_geometry):
        """Encode a ``[M,8]`` normalized geometry tensor."""
        return self.network(raw_geometry)


class SharedGeometryRefiner(nn.Module):
    """Decode object and proposal embeddings into periodic 8D residuals."""

    def __init__(self, embedding_dim, geometry_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, object_embedding, geometry_embedding):
        """Predict residuals for matching object and geometry embeddings."""
        if object_embedding.shape[0] != geometry_embedding.shape[0]:
            raise ValueError("object and geometry embedding counts must match")
        return self.network(torch.cat((object_embedding, geometry_embedding), dim=-1))


class SharedMultiScaleFusion(nn.Module):
    """Fuse detail/context embeddings while starting as exact detail-only."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, detail_embedding, context_embedding):
        """Return ``[M,D]`` residual-safe concat-projection fusion."""
        if detail_embedding.shape != context_embedding.shape:
            raise ValueError("detail and context embeddings must have equal shape")
        projected = self.projection(
            torch.cat((detail_embedding, context_embedding), dim=-1)
        )
        return detail_embedding + self.residual_scale * projected


class SharedAdaptiveScaleGate(nn.Module):
    """Predict one shared detail-vs-context gate per agent-object pair."""

    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = max(16, embedding_dim // 2)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, 4.0)

    def forward(self, detail_embedding, context_embedding):
        """Return fused embeddings and scalar gates in ``[M,1]``."""
        if detail_embedding.shape != context_embedding.shape:
            raise ValueError("detail and context embeddings must have equal shape")
        gate = torch.sigmoid(
            self.network(torch.cat((detail_embedding, context_embedding), dim=-1))
        )
        return (
            gate * detail_embedding + (1.0 - gate) * context_embedding,
            gate,
        )


class SharedObjectQualityHead(nn.Module):
    """Predict agent-object-specific scalar geometry quality in ``[0,1]``."""

    def __init__(
        self,
        embedding_dim,
        geometry_dim,
        hidden_dim,
        use_roi_coverage,
        use_agent_distance,
    ):
        super().__init__()
        self.use_roi_coverage = bool(use_roi_coverage)
        self.use_agent_distance = bool(use_agent_distance)
        input_dim = embedding_dim + geometry_dim
        input_dim += int(self.use_roi_coverage) + int(self.use_agent_distance)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        object_embedding,
        geometry_embedding,
        roi_coverage=None,
        agent_distance=None,
    ):
        """Return scalar quality for matching valid agent-object pairs."""
        values = [object_embedding, geometry_embedding]
        if self.use_roi_coverage:
            values.append(_quality_scalar(roi_coverage, object_embedding, "roi_coverage"))
        if self.use_agent_distance:
            values.append(_quality_scalar(agent_distance, object_embedding, "agent_distance"))
        return torch.sigmoid(self.network(torch.cat(values, dim=-1))).squeeze(-1)


def dual_space_is_enabled(args):
    """Return the explicit DS-V1 enable flag while rejecting ambiguous values."""
    config = args.get("dual_space")
    if config is None:
        return False
    if not isinstance(config, dict):
        raise TypeError("model.args.dual_space must be a mapping")
    validate_dual_space_config(config)
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("dual_space.enabled must be bool")
    return enabled


def install_dual_space_modules(model, args):
    """Install only the explicitly enabled Dual-Space parameter groups."""
    model.dual_space_enabled = dual_space_is_enabled(args)
    model._dual_space_log_printed = False
    if not model.dual_space_enabled:
        print("[DualSpace] enabled=False")
        return

    config = validate_dual_space_config(args["dual_space"])
    if (
        config["mode"] == "stage2_adapt"
        and config["active_modality"] not in model.modality_name_list
    ):
        raise ValueError(
            "dual_space.active_modality must be present in the Stage2 model"
        )
    channels = infer_common_bev_channels(args, model.modality_name_list)
    geometry = DualSpaceBEVGeometry.from_lidar_range(args["lidar_range"])
    roi_config = config["roi"]
    encoder_config = config["object_encoder"]
    geometry_config = config["geometry_encoder"]
    refiner_config = config["refiner"]

    model.dual_space_config = config
    model.dual_space_flags = dual_space_feature_flags(config)
    model._dual_space_checkpoint_ready = bool(
        config["mode"] == "stage1_anchor"
        and config["allow_untrained_initialization"]
    )
    model.dual_space_common_bev_channels = channels
    model.dual_space_bev_geometry = geometry
    detail_output_size = (
        config["multi_scale"]["detail"]["roi_size"]
        if model.dual_space_flags["multi_scale"]
        else roi_config["output_size"]
    )
    model.dual_space_object_roi = ChunkedRotatedBEVROISampler(
        bev_geometry=geometry,
        output_size=detail_output_size,
        chunk_size=roi_config["chunk_size"],
        min_coverage=roi_config["min_coverage"],
    )
    model.dual_space_training_proposal_sampler = DualSpaceTrainingProposalSampler(
        config["training_proposals"],
        max_proposals=config["training_proposals"]["max_proposals"],
    )
    for modality in model.modality_name_list:
        if modality == "m1":
            adapter = nn.Identity()
        else:
            adapter = ResidualObjectAdapter(channels)
        setattr(model, "dual_space_object_adapter_%s" % modality, adapter)

    model.dual_space_shared_object_encoder = SharedObjectEncoder(
        in_channels=channels,
        hidden_channels=encoder_config["hidden_channels"],
        pooled_size=encoder_config["pooled_size"],
        embedding_dim=encoder_config["embedding_dim"],
    )
    model.dual_space_shared_geometry_encoder = SharedGeometryEncoder(
        geometry_config["hidden_dim"]
    )
    model.dual_space_shared_object_refiner = SharedGeometryRefiner(
        embedding_dim=encoder_config["embedding_dim"],
        geometry_dim=geometry_config["hidden_dim"],
        hidden_dim=refiner_config["hidden_dim"],
    )

    if model.dual_space_flags["multi_scale"]:
        context_channels = infer_context_bev_channels(args)
        multi_config = config["multi_scale"]
        context_config = multi_config["context"]
        model.dual_space_context_bev_channels = context_channels
        model.dual_space_context_roi = ChunkedRotatedBEVROISampler(
            bev_geometry=geometry,
            output_size=context_config["roi_size"],
            chunk_size=roi_config["chunk_size"],
            min_coverage=roi_config["min_coverage"],
        )
        for modality in model.modality_name_list:
            adapter = (
                nn.Identity()
                if modality == "m1"
                else ResidualObjectAdapter(context_channels)
            )
            setattr(model, "dual_space_context_adapter_%s" % modality, adapter)
        model.dual_space_shared_context_encoder = SharedObjectEncoder(
            in_channels=context_channels,
            hidden_channels=encoder_config["hidden_channels"],
            pooled_size=encoder_config["pooled_size"],
            embedding_dim=encoder_config["embedding_dim"],
        )
        if multi_config["fusion"] == "concat_projection":
            model.dual_space_shared_multiscale_fusion = SharedMultiScaleFusion(
                encoder_config["embedding_dim"]
            )
        else:
            model.dual_space_shared_scale_gate = SharedAdaptiveScaleGate(
                encoder_config["embedding_dim"]
            )

    if model.dual_space_flags["quality"]:
        quality_config = config["quality"]
        model.dual_space_shared_quality_head = SharedObjectQualityHead(
            embedding_dim=encoder_config["embedding_dim"],
            geometry_dim=geometry_config["hidden_dim"],
            hidden_dim=quality_config["hidden_dim"],
            use_roi_coverage=quality_config["use_roi_coverage"],
            use_agent_distance=quality_config["use_agent_distance"],
        )


def configure_dual_space_trainability(model):
    """Apply Stage1/Stage2 trainability and frozen shared-module modes."""
    if not getattr(model, "dual_space_enabled", False):
        return
    mode = model.dual_space_config["mode"]
    shared_trainable = mode == "stage1_anchor"
    shared_modules = [
        model.dual_space_shared_object_encoder,
        model.dual_space_shared_geometry_encoder,
        model.dual_space_shared_object_refiner,
    ]
    for name in (
        "dual_space_shared_context_encoder",
        "dual_space_shared_multiscale_fusion",
        "dual_space_shared_scale_gate",
        "dual_space_shared_quality_head",
    ):
        if hasattr(model, name):
            shared_modules.append(getattr(model, name))
    for module in shared_modules:
        _set_module_trainability(module, shared_trainable, model.training)

    active_modality = model.dual_space_config.get("active_modality")
    for modality in model.modality_name_list:
        adapter = getattr(model, "dual_space_object_adapter_%s" % modality)
        adapter_trainable = mode == "stage2_adapt" and modality == active_modality
        _set_module_trainability(adapter, adapter_trainable, model.training)
        context_name = "dual_space_context_adapter_%s" % modality
        if hasattr(model, context_name):
            _set_module_trainability(
                getattr(model, context_name), adapter_trainable, model.training
            )

    if not model._dual_space_log_printed:
        _print_dual_space_summary(model, shared_trainable, active_modality)
        model._dual_space_log_printed = True


def configure_dual_space_proposal_decoder(model, hypes):
    """Attach the parameter-free HEAL decoder only for mixed/RPR profiles."""
    if not getattr(model, "dual_space_enabled", False):
        return
    flags = model.dual_space_flags
    if not (flags["mixed_proposals"] or flags["remote_proposal_rescue"]):
        return
    postprocess = hypes.get("postprocess")
    if not isinstance(postprocess, dict):
        raise TypeError(
            "mixed proposals/RPR require the top-level postprocess mapping"
        )
    if postprocess.get("order", "hwl") != "hwl":
        raise ValueError(
            "Dual-Space detector proposal decoding expects postprocess.order=hwl"
        )
    dir_args = postprocess.get(
        "dir_args", hypes["model"]["args"].get("dir_args", {})
    )
    model.dual_space_detector_proposal_decoder = DualSpaceDetectorProposalDecoder(
        postprocess,
        dir_args,
        hypes["model"]["args"]["lidar_range"],
    )


def build_collab_dual_space_context(
    model,
    agent_features,
    record_len,
    affine_matrix,
    agent_modality_list,
    pairwise_t_matrix=None,
):
    """Warp per-agent Common-BEV maps to ego while retaining agent identity."""
    if not getattr(model, "dual_space_enabled", False):
        return None
    if agent_features.ndim != 4:
        raise ValueError("agent_features must have shape [sum(A),C,H,W]")
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    if sum(lengths) != int(agent_features.shape[0]):
        raise ValueError("record_len does not match Common-BEV agent count")
    if len(agent_modality_list) != int(agent_features.shape[0]):
        raise ValueError("agent_modality_list does not match Common-BEV agent count")

    source_support = _build_source_support(model, agent_features, agent_modality_list)
    scenes = []
    offset = 0
    _, channels, height, width = agent_features.shape
    for batch_index, agent_count in enumerate(lengths):
        if agent_count < 1:
            raise ValueError("every scene must contain at least one agent")
        scene_features = agent_features[offset:offset + agent_count]
        scene_support = source_support[offset:offset + agent_count]
        transforms = affine_matrix[
            batch_index, 0, :agent_count
        ].to(device=agent_features.device, dtype=agent_features.dtype)
        grid = F.affine_grid(
            transforms,
            (agent_count, channels, height, width),
            align_corners=False,
        )
        aligned_features = F.grid_sample(
            scene_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        aligned_support = F.grid_sample(
            scene_support,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        scene = {
            "agent_features": aligned_features,
            "agent_support": aligned_support,
            "agent_modalities": tuple(
                agent_modality_list[offset:offset + agent_count]
            ),
        }
        if (
            model.dual_space_flags["quality"]
            and model.dual_space_config["quality"]["use_agent_distance"]
        ):
            if pairwise_t_matrix is None:
                raise ValueError(
                    "quality-aware Dual-Space context requires pairwise_t_matrix"
                )
            scene["agent_positions"] = _agent_positions_in_ego(
                pairwise_t_matrix[batch_index, 0, :agent_count],
                agent_features,
            )
        scenes.append(scene)
        offset += agent_count
    return {"scenes": tuple(scenes), "box_order": "hwl", "aligned_to": "ego"}


def build_single_dual_space_context(model, feature, modality_name):
    """Build one-agent-per-sample ego-aligned contexts for HEAL Stage2."""
    if not getattr(model, "dual_space_enabled", False):
        return None
    if modality_name not in model.modality_name_list:
        raise ValueError("unknown active modality %s" % modality_name)
    modalities = [modality_name] * int(feature.shape[0])
    support = _build_source_support(model, feature, modalities)
    scenes = []
    for batch_index in range(int(feature.shape[0])):
        scene = {
            "agent_features": feature[batch_index:batch_index + 1],
            "agent_support": support[batch_index:batch_index + 1],
            "agent_modalities": (modality_name,),
        }
        if (
            model.dual_space_flags["quality"]
            and model.dual_space_config["quality"]["use_agent_distance"]
        ):
            scene["agent_positions"] = feature.new_zeros((1, 2))
        scenes.append(scene)
    return {"scenes": tuple(scenes), "box_order": "hwl", "aligned_to": "ego"}


def attach_collab_dual_space_pyramid_context(
    model,
    context,
    pre_fusion_features,
    record_len,
    affine_matrix,
    agent_modality_list,
):
    """Attach only enabled pre-fusion common-pyramid levels to scene context."""
    if context is None:
        return None
    need_multi = model.dual_space_flags["multi_scale"]
    need_remote = model.dual_space_flags["remote_proposal_rescue"]
    if not (need_multi or need_remote):
        return context
    if not isinstance(pre_fusion_features, (tuple, list)):
        raise TypeError("pre_fusion_features must be a tuple/list")
    if need_multi and len(pre_fusion_features) < 2:
        raise RuntimeError("multi-scale Dual-Space requires pyramid level 1")
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    capture_indices = range(len(pre_fusion_features)) if need_remote else (1,)
    aligned_by_level = {}
    support_by_level = {}
    for level_index in capture_indices:
        feature = pre_fusion_features[level_index]
        if int(feature.shape[0]) != sum(lengths):
            raise ValueError("pyramid feature agent count does not match record_len")
        source_support = _build_source_support(model, feature, agent_modality_list)
        aligned_scenes = []
        aligned_support_scenes = []
        offset = 0
        for batch_index, agent_count in enumerate(lengths):
            transforms = affine_matrix[
                batch_index, 0, :agent_count
            ].to(device=feature.device, dtype=feature.dtype)
            scene_feature = feature[offset:offset + agent_count]
            scene_support = source_support[offset:offset + agent_count]
            aligned_scenes.append(
                _warp_scene_features(scene_feature, transforms, "bilinear")
            )
            aligned_support_scenes.append(
                _warp_scene_features(scene_support, transforms, "nearest")
            )
            offset += agent_count
        aligned_by_level[level_index] = tuple(aligned_scenes)
        support_by_level[level_index] = tuple(aligned_support_scenes)

    for scene_index, scene in enumerate(context["scenes"]):
        if need_multi:
            scene["context_agent_features"] = aligned_by_level[1][scene_index]
            scene["context_agent_support"] = support_by_level[1][scene_index]
        if need_remote:
            scene["aligned_pyramid_features"] = tuple(
                aligned_by_level[index][scene_index]
                for index in range(len(pre_fusion_features))
            )
    return context


def attach_single_dual_space_pyramid_context(
    model, context, pre_fusion_features, modality_name
):
    """Attach Stage2 single-agent common-pyramid context without spatial warp."""
    if context is None or not model.dual_space_flags["multi_scale"]:
        return context
    if not isinstance(pre_fusion_features, (tuple, list)) or len(pre_fusion_features) < 2:
        raise RuntimeError("multi-scale Dual-Space requires pyramid level 1")
    context_feature = pre_fusion_features[1]
    support = _build_source_support(
        model,
        context_feature,
        [modality_name] * int(context_feature.shape[0]),
    )
    if len(context["scenes"]) != int(context_feature.shape[0]):
        raise ValueError("single pyramid feature batch does not match context")
    for batch_index, scene in enumerate(context["scenes"]):
        scene["context_agent_features"] = context_feature[
            batch_index:batch_index + 1
        ]
        scene["context_agent_support"] = support[batch_index:batch_index + 1]
    return context


@torch.no_grad()
def attach_remote_detector_proposals(model, context, data_dict):
    """Decode per-agent ego-aligned pyramid features for DS-V4 rescue."""
    if not model.dual_space_flags["remote_proposal_rescue"]:
        return context
    decoder = getattr(model, "dual_space_detector_proposal_decoder", None)
    if decoder is None:
        raise RuntimeError("RPR requires a configured detector proposal decoder")
    anchor_box = data_dict.get("anchor_box")
    if not torch.is_tensor(anchor_box):
        raise KeyError("RPR requires data_dict['anchor_box']")
    for scene_index, scene in enumerate(context["scenes"]):
        levels = scene.get("aligned_pyramid_features")
        if not isinstance(levels, tuple):
            raise RuntimeError("RPR context is missing aligned pyramid features")
        decoded_feature = model.pyramid_backbone.decode_multiscale_feature(
            list(levels)
        )
        if model.shrink_flag:
            decoded_feature = model.shrink_conv(decoded_feature)
        cls_preds = model.cls_head(decoded_feature)
        reg_preds = model.reg_head(decoded_feature)
        dir_preds = model.dir_head(decoded_feature)
        agent_count = int(decoded_feature.shape[0])
        anchors = _scene_anchor_for_decoder(
            anchor_box, scene_index, len(context["scenes"])
        )
        proposals, scores = decoder.decode(
            cls_preds, reg_preds, dir_preds, anchors
        )
        scene["remote_proposals"] = proposals
        scene["remote_scores"] = scores
        # Dense maps are intentionally not retained after detached decoding.
        del scene["aligned_pyramid_features"]
    return context


@torch.no_grad()
def decode_dual_space_detector_output(model, detector_output, data_dict):
    """Decode fused HEAL predictions for optional mixed proposal training."""
    decoder = getattr(model, "dual_space_detector_proposal_decoder", None)
    if decoder is None:
        raise RuntimeError(
            "mixed proposal training requires a configured detector decoder"
        )
    if not isinstance(detector_output, dict):
        raise TypeError("detector_output must be a mapping")
    anchor_box = data_dict.get("anchor_box")
    if not torch.is_tensor(anchor_box):
        raise KeyError("mixed proposal training requires data_dict['anchor_box']")
    return decoder.decode(
        detector_output["cls_preds"],
        detector_output["reg_preds"],
        detector_output["dir_preds"],
        anchor_box,
    )


def run_dual_space_training(model, context, data_dict, detector_output=None):
    """Run configured object supervision for Stage1 or independent Stage2."""
    if not getattr(model, "dual_space_enabled", False):
        return None
    assert_dual_space_runtime_ready(model)
    mode = model.dual_space_config["mode"]
    if mode not in ("stage1_anchor", "stage2_adapt"):
        return None
    if "object_bbx_center" not in data_dict or "object_bbx_mask" not in data_dict:
        raise KeyError("dual-space training requires object_bbx_center and object_bbx_mask")
    gt_boxes = data_dict["object_bbx_center"]
    gt_mask = data_dict["object_bbx_mask"]
    if gt_boxes.ndim != 3 or gt_boxes.shape[-1] != 7:
        raise ValueError("object_bbx_center must have shape [B,M,7] in hwl order")
    if gt_boxes.shape[:2] != gt_mask.shape[:2]:
        raise ValueError("object_bbx_mask must match object_bbx_center [B,M]")
    if len(context["scenes"]) != int(gt_boxes.shape[0]):
        raise ValueError("dual-space context scene count does not match GT batch")

    if mode == "stage2_adapt":
        # GroupNorm/LayerNorm have no running statistics, but explicit eval mode
        # protects the frozen shared-space contract from future module changes.
        model.dual_space_shared_object_encoder.eval()
        model.dual_space_shared_geometry_encoder.eval()
        model.dual_space_shared_object_refiner.eval()

    predicted_proposals = predicted_scores = None
    if model.dual_space_flags["mixed_proposals"]:
        predicted_proposals, predicted_scores = decode_dual_space_detector_output(
            model, detector_output, data_dict
        )
        if len(predicted_proposals) != len(context["scenes"]):
            raise ValueError("decoded detector batch does not match context scenes")

    scene_outputs = []
    total_pairs = 0
    valid_pairs = 0
    coverage_sum = gt_boxes.new_zeros(())
    proposal_count = 0
    for scene_index, scene in enumerate(context["scenes"]):
        if model.dual_space_flags["mixed_proposals"]:
            proposals, targets = model.dual_space_training_proposal_sampler(
                gt_boxes[scene_index],
                gt_mask[scene_index],
                with_jitter=bool(model.training),
                predicted_boxes=predicted_proposals[scene_index],
                predicted_scores=predicted_scores[scene_index],
            )
        else:
            # Keep the original DS-V1 call signature and RNG path unchanged.
            proposals, targets = model.dual_space_training_proposal_sampler(
                gt_boxes[scene_index],
                gt_mask[scene_index],
                with_jitter=bool(model.training),
            )
        result = predict_scene_residuals(model, scene, proposals)
        target_residuals = encode_box_residual(proposals, targets)
        pair_indices = result["valid_mask"].nonzero(as_tuple=False)
        individual_targets = target_residuals.index_select(0, pair_indices[:, 0]) \
            if pair_indices.numel() else target_residuals.new_empty((0, 8))
        result.update(
            {
                "targets": targets,
                "target_residuals": target_residuals,
                "individual_targets": individual_targets,
            }
        )
        if model.dual_space_flags["quality"]:
            if pair_indices.numel():
                selected_proposals = proposals.index_select(0, pair_indices[:, 0])
                selected_targets = targets.index_select(0, pair_indices[:, 0])
                individual_boxes = decode_box_residual(
                    selected_proposals, result["individual_residuals"]
                )
                quality_targets = aligned_rotated_bev_iou_hwl(
                    individual_boxes.detach(), selected_targets.detach()
                )
            else:
                quality_targets = proposals.new_empty((0,))
            result["quality_targets"] = quality_targets.detach()
        scene_outputs.append(result)
        proposal_count += int(proposals.shape[0])
        total_pairs += int(result["valid_mask"].numel())
        valid_pairs += int(result["valid_mask"].sum().item())
        coverage_sum = coverage_sum + result["coverage"].detach().sum()

    denominator = max(total_pairs, 1)
    stats = {
        "object_roi_count": proposal_count,
        "valid_agent_object_pairs": valid_pairs,
        "valid_object_ratio": float(valid_pairs) / float(denominator),
        "mean_roi_coverage": float(coverage_sum.item()) / float(denominator),
    }
    payload = {
        "enabled": True,
        "version": model.dual_space_config["version"],
        "mode": mode,
        "scenes": tuple(scene_outputs),
        "loss_config": dict(model.dual_space_config["loss"]),
        "stats": stats,
    }
    if model.dual_space_flags["quality"]:
        payload["quality_enabled"] = True
    return payload


def predict_scene_residuals(model, scene, proposals):
    """Predict per-agent residuals and configured geometry consensus."""
    assert_dual_space_runtime_ready(model)
    agent_features = scene["agent_features"]
    roi_features, detail_valid, detail_coverage = model.dual_space_object_roi(
        agent_features,
        proposals,
        scene.get("agent_support"),
    )
    valid_mask = detail_valid
    coverage = detail_coverage
    context_roi_features = None
    context_coverage = None
    if model.dual_space_flags["multi_scale"]:
        if "context_agent_features" not in scene:
            raise RuntimeError(
                "multi-scale Dual-Space scene is missing context pyramid features"
            )
        context_roi_features, context_valid, context_coverage = (
            model.dual_space_context_roi(
                scene["context_agent_features"],
                proposals,
                scene.get("context_agent_support"),
            )
        )
        valid_mask = detail_valid & context_valid
        coverage = torch.minimum(detail_coverage, context_coverage)

    proposal_count, agent_count = valid_mask.shape
    valid_indices = valid_mask.nonzero(as_tuple=False)
    scale_gates = None
    individual_quality = None
    if valid_indices.numel():
        proposal_indices = valid_indices[:, 0]
        agent_indices = valid_indices[:, 1]
        selected_rois = roi_features[proposal_indices, agent_indices]
        selected_modalities = [
            scene["agent_modalities"][int(index)] for index in agent_indices.tolist()
        ]
        adapted_rois = route_modality_adapters(
            model, selected_rois, selected_modalities
        )
        detail_embedding = model.dual_space_shared_object_encoder(adapted_rois)
        object_embedding = detail_embedding
        if model.dual_space_flags["multi_scale"]:
            selected_context_rois = context_roi_features[
                proposal_indices, agent_indices
            ]
            adapted_context_rois = route_modality_adapters(
                model,
                selected_context_rois,
                selected_modalities,
                adapter_namespace="context_adapter",
            )
            context_embedding = model.dual_space_shared_context_encoder(
                adapted_context_rois
            )
            fusion_mode = model.dual_space_config["multi_scale"]["fusion"]
            if fusion_mode == "concat_projection":
                object_embedding = model.dual_space_shared_multiscale_fusion(
                    detail_embedding, context_embedding
                )
            else:
                object_embedding, scale_gates = (
                    model.dual_space_shared_scale_gate(
                        detail_embedding, context_embedding
                    )
                )
        selected_proposals = proposals.index_select(0, proposal_indices)
        raw_geometry = proposal_geometry_raw(
            selected_proposals, model.dual_space_bev_geometry
        )
        raw_geometry = raw_geometry.to(
            device=object_embedding.device,
            dtype=object_embedding.dtype,
        )
        geometry_embedding = model.dual_space_shared_geometry_encoder(raw_geometry)
        individual_residuals = model.dual_space_shared_object_refiner(
            object_embedding, geometry_embedding
        )
        per_agent_residuals = individual_residuals.new_zeros(
            (proposal_count, agent_count, 8)
        ).index_put(
            (proposal_indices, agent_indices), individual_residuals
        )
        if model.dual_space_flags["quality"]:
            normalized_distances = None
            if model.dual_space_config["quality"]["use_agent_distance"]:
                normalized_distances = normalized_agent_object_distance(
                    scene, proposals, model.dual_space_bev_geometry
                )
            individual_quality = model.dual_space_shared_quality_head(
                object_embedding,
                geometry_embedding,
                roi_coverage=coverage[proposal_indices, agent_indices],
                agent_distance=(
                    normalized_distances[proposal_indices, agent_indices]
                    if normalized_distances is not None
                    else None
                ),
            )
            per_agent_quality = individual_quality.new_zeros(
                (proposal_count, agent_count)
            ).index_put(
                (proposal_indices, agent_indices), individual_quality
            )
    else:
        individual_residuals = agent_features.new_empty((0, 8))
        per_agent_residuals = agent_features.new_zeros(
            (proposal_count, agent_count, 8)
        )
        if model.dual_space_flags["quality"]:
            individual_quality = agent_features.new_empty((0,))
            per_agent_quality = agent_features.new_zeros(
                (proposal_count, agent_count)
            )

    if model.dual_space_config["consensus"]["mode"] == "quality_weighted":
        quality_config = model.dual_space_config["quality"]
        fused_residuals, any_valid, consensus_weights, quality_fallback = (
            quality_weighted_geometry_consensus(
                per_agent_residuals,
                valid_mask,
                per_agent_quality,
                min_quality_sum=model.dual_space_config["consensus"][
                    "min_quality_sum"
                ],
                detach_quality=quality_config["detach_weight_for_consensus"],
            )
        )
    else:
        fused_residuals, any_valid = uniform_geometry_consensus(
            per_agent_residuals, valid_mask
        )
        consensus_weights = quality_fallback = None
    decoded = decode_box_residual(proposals, fused_residuals)
    refined_boxes = torch.where(any_valid[:, None], decoded, proposals)
    result = {
        "proposals": proposals,
        "individual_residuals": individual_residuals,
        "per_agent_residuals": per_agent_residuals,
        "fused_residuals": fused_residuals,
        "refined_boxes": refined_boxes,
        "valid_mask": valid_mask,
        "any_valid": any_valid,
        "coverage": coverage,
    }
    if model.dual_space_flags["multi_scale"]:
        result.update(
            {
                "detail_coverage": detail_coverage,
                "context_coverage": context_coverage,
                "scale_gates": scale_gates,
            }
        )
    if model.dual_space_flags["quality"]:
        result.update(
            {
                "individual_quality": individual_quality,
                "per_agent_quality": per_agent_quality,
                "consensus_weights": consensus_weights,
                "quality_fallback": quality_fallback,
            }
        )
    return result


def route_modality_adapters(
    model, roi_features, modality_names, adapter_namespace="object_adapter"
):
    """Route valid ROIs by real modality labels, never by agent position."""
    if roi_features.shape[0] != len(modality_names):
        raise ValueError("modality_names must match ROI count")
    if roi_features.shape[0] == 0:
        return roi_features
    grouped_indices = OrderedDict()
    for index, modality in enumerate(modality_names):
        grouped_indices.setdefault(modality, []).append(index)

    output_parts = []
    position_parts = []
    for modality, indices in grouped_indices.items():
        attribute = "dual_space_%s_%s" % (adapter_namespace, modality)
        if not hasattr(model, attribute):
            raise KeyError("no dual-space object adapter for modality %s" % modality)
        positions = torch.tensor(indices, dtype=torch.long, device=roi_features.device)
        selected = roi_features.index_select(0, positions)
        output_parts.append(getattr(model, attribute)(selected))
        position_parts.append(positions)
    packed_outputs = torch.cat(output_parts, dim=0)
    packed_positions = torch.cat(position_parts, dim=0)
    inverse_order = torch.argsort(packed_positions)
    return packed_outputs.index_select(0, inverse_order)


def proposal_geometry_raw(proposals, geometry):
    """Build ``[x_norm,y_norm,z_norm,log(l,w,h),sin(yaw),cos(yaw)]``."""
    if not isinstance(geometry, DualSpaceBEVGeometry):
        raise TypeError("geometry must be DualSpaceBEVGeometry")
    if not torch.is_tensor(proposals) or proposals.ndim != 2 or proposals.shape[1] != 7:
        raise ValueError("proposals must have shape [N,7] in hwl order")
    if not torch.is_floating_point(proposals):
        raise TypeError("proposals must use a floating-point dtype")
    if proposals.numel() and not bool((proposals[:, 3:6] > 0).all()):
        raise ValueError("proposal height, width, and length must be positive")
    x_norm = 2.0 * (proposals[:, 0] - geometry.x_min) / (
        geometry.x_max - geometry.x_min
    ) - 1.0
    y_norm = 2.0 * (proposals[:, 1] - geometry.y_min) / (
        geometry.y_max - geometry.y_min
    ) - 1.0
    z_norm = 2.0 * (proposals[:, 2] - geometry.z_min) / (
        geometry.z_max - geometry.z_min
    ) - 1.0
    return torch.stack(
        (
            x_norm,
            y_norm,
            z_norm,
            torch.log(proposals[:, 5]),
            torch.log(proposals[:, 4]),
            torch.log(proposals[:, 3]),
            torch.sin(proposals[:, 6]),
            torch.cos(proposals[:, 6]),
        ),
        dim=-1,
    )


def uniform_geometry_consensus(per_agent_residuals, valid_mask):
    """Average valid decoded-geometry residual components uniformly."""
    if per_agent_residuals.ndim != 3 or per_agent_residuals.shape[-1] != 8:
        raise ValueError("per_agent_residuals must have shape [N,A,8]")
    if tuple(valid_mask.shape) != tuple(per_agent_residuals.shape[:2]):
        raise ValueError("valid_mask must have shape [N,A]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    weights = valid_mask.to(dtype=per_agent_residuals.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1)
    any_valid = counts[:, 0] > 0
    fused = (per_agent_residuals * weights).sum(dim=1) / counts.clamp_min(1.0)
    fused = torch.where(any_valid[:, None], fused, torch.zeros_like(fused))
    return fused, any_valid


def quality_weighted_geometry_consensus(
    per_agent_residuals,
    valid_mask,
    per_agent_quality,
    min_quality_sum=1e-6,
    detach_quality=True,
):
    """Fuse residuals with scalar quality and deterministic uniform fallback."""
    if per_agent_residuals.ndim != 3 or per_agent_residuals.shape[-1] != 8:
        raise ValueError("per_agent_residuals must have shape [N,A,8]")
    if tuple(valid_mask.shape) != tuple(per_agent_residuals.shape[:2]):
        raise ValueError("valid_mask must have shape [N,A]")
    if tuple(per_agent_quality.shape) != tuple(valid_mask.shape):
        raise ValueError("per_agent_quality must have shape [N,A]")
    if type(detach_quality) is not bool:
        raise TypeError("detach_quality must be bool")
    min_quality_sum = float(min_quality_sum)
    if min_quality_sum <= 0.0:
        raise ValueError("min_quality_sum must be positive")

    quality = per_agent_quality.detach() if detach_quality else per_agent_quality
    valid = valid_mask.to(dtype=per_agent_residuals.dtype)
    quality_weights = quality.clamp(0.0, 1.0) * valid
    quality_sum = quality_weights.sum(dim=1, keepdim=True)
    valid_count = valid.sum(dim=1, keepdim=True)
    any_valid = valid_count[:, 0] > 0
    fallback = any_valid & (quality_sum[:, 0] < min_quality_sum)
    normalized_quality = quality_weights / quality_sum.clamp_min(min_quality_sum)
    uniform_weights = valid / valid_count.clamp_min(1.0)
    weights = torch.where(
        fallback[:, None], uniform_weights, normalized_quality
    )
    weights = torch.where(any_valid[:, None], weights, torch.zeros_like(weights))
    quality_fused = (
        per_agent_residuals * normalized_quality.unsqueeze(-1)
    ).sum(dim=1)
    uniform_fused, _ = uniform_geometry_consensus(
        per_agent_residuals, valid_mask
    )
    fused = torch.where(fallback[:, None], uniform_fused, quality_fused)
    fused = torch.where(any_valid[:, None], fused, torch.zeros_like(fused))
    return fused, any_valid, weights, fallback


def normalized_agent_object_distance(scene, proposals, geometry):
    """Return ``[P,A]`` object-agent distance normalized by BEV diagonal."""
    positions = scene.get("agent_positions")
    if not torch.is_tensor(positions) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("quality-aware scene requires agent_positions [A,2]")
    if positions.device != proposals.device:
        positions = positions.to(device=proposals.device, dtype=proposals.dtype)
    else:
        positions = positions.to(dtype=proposals.dtype)
    diagonal = (
        (geometry.x_max - geometry.x_min) ** 2
        + (geometry.y_max - geometry.y_min) ** 2
    ) ** 0.5
    delta = proposals[:, None, :2] - positions[None, :, :]
    return torch.linalg.vector_norm(delta, dim=-1).div(diagonal).clamp(0.0, 1.0)


def refine_dual_space_detections(model, pred_box_tensor, pred_score, context):
    """Optionally rescue, then refine top-K boxes while preserving all scores."""
    if not getattr(model, "dual_space_enabled", False):
        return pred_box_tensor, pred_score
    if model.dual_space_config["mode"] != "inference":
        raise RuntimeError("dual-space detection refinement requires mode=inference")
    if context is None or len(context.get("scenes", ())) != 1:
        raise RuntimeError("dual-space inference requires one same-forward scene context")
    scene = context["scenes"][0]
    rescue_enabled = model.dual_space_flags["remote_proposal_rescue"]
    report_stats = model.dual_space_flags["report_stats"]
    if (pred_box_tensor is None) != (pred_score is None):
        raise ValueError(
            "prediction boxes and scores must either both be tensors or both be None"
        )
    if pred_box_tensor is None or pred_score is None:
        if not rescue_enabled:
            return pred_box_tensor, pred_score
        remote_boxes = scene.get("remote_proposals")
        remote_scores = scene.get("remote_scores")
        if not remote_boxes:
            return pred_box_tensor, pred_score
        reference_box = remote_boxes[0]
        reference_score = remote_scores[0]
        center_boxes = reference_box.new_empty((0, 7))
        pred_score = reference_score.new_empty((0,))
    else:
        if pred_box_tensor.shape[0] != pred_score.shape[0]:
            raise ValueError("prediction box and score counts must match")
        center_boxes = corners_3d_to_boxes_hwl(pred_box_tensor)

    original_fused_count = int(center_boxes.shape[0])
    runtime_stats = None
    if report_stats:
        runtime_stats = {
            "original_fused_proposals": original_fused_count,
            "rescued_remote_proposals": 0,
            "final_candidate_count": original_fused_count,
            "refined_proposal_count": 0,
            "valid_agent_object_pairs": 0,
            "mean_agents_per_object": 0.0,
            "mean_roi_coverage": 0.0,
        }
    if rescue_enabled:
        remote_boxes = scene.get("remote_proposals")
        remote_scores = scene.get("remote_scores")
        if remote_boxes is None or remote_scores is None:
            raise RuntimeError("RPR inference context is missing remote proposals")
        center_boxes, pred_score, rescue_stats = rescue_remote_proposals(
            center_boxes,
            pred_score,
            remote_boxes,
            remote_scores,
            model.dual_space_config["remote_proposal_rescue"],
        )
        if report_stats:
            runtime_stats.update(rescue_stats)
            runtime_stats["rescued_remote_proposals"] = rescue_stats[
                "rescued_proposal_count"
            ]
            runtime_stats["final_candidate_count"] = rescue_stats[
                "candidate_proposal_count"
            ]

    if center_boxes.shape[0] == 0:
        if report_stats:
            context["dual_space_stats"] = runtime_stats
        if not rescue_enabled:
            return pred_box_tensor, pred_score
        return boxes_hwl_to_corners_3d(center_boxes), pred_score

    max_count = model.dual_space_config["roi"]["max_infer_proposals"]
    refine_count = min(int(center_boxes.shape[0]), max_count)
    top_indices = torch.topk(pred_score, k=refine_count, sorted=False).indices
    selected = center_boxes.index_select(0, top_indices).detach()
    result = predict_scene_residuals(model, scene, selected)
    selected_refined = torch.where(
        result["any_valid"][:, None], result["refined_boxes"], selected
    )
    refined_corners = boxes_hwl_to_corners_3d(selected_refined)

    if rescue_enabled:
        rescued_corners = boxes_hwl_to_corners_3d(
            center_boxes[original_fused_count:]
        )
        output_boxes = (
            rescued_corners
            if pred_box_tensor is None
            else torch.cat((pred_box_tensor.clone(), rescued_corners), dim=0)
        )
    else:
        output_boxes = pred_box_tensor.clone()
    valid_top_indices = top_indices[result["any_valid"]]
    if valid_top_indices.numel():
        output_boxes[valid_top_indices] = refined_corners[result["any_valid"]]
    if report_stats:
        refined_count = int(result["any_valid"].sum().item())
        valid_pair_count = int(result["valid_mask"].sum().item())
        runtime_stats.update(
            {
                "refined_proposal_count": refined_count,
                "refined_candidate_count": refined_count,
                "valid_agent_object_pairs": valid_pair_count,
                "valid_roi_ratio": float(
                    result["valid_mask"].float().mean().item()
                ) if result["valid_mask"].numel() else 0.0,
                "mean_roi_coverage": float(result["coverage"].mean().item())
                if result["coverage"].numel() else 0.0,
                "mean_agents_per_object": float(
                    result["valid_mask"].sum(dim=1).float().mean().item()
                ) if result["valid_mask"].shape[0] else 0.0,
            }
        )
        runtime_stats["mean_contributing_agents"] = runtime_stats[
            "mean_agents_per_object"
        ]
        if (
            model.dual_space_flags["quality"]
            and result["individual_quality"].numel()
        ):
            quality = result["individual_quality"].detach()
            runtime_stats.update(
                {
                    "mean_quality": float(quality.mean().item()),
                    "median_quality": float(quality.median().item()),
                    "low_quality_fraction": float(
                        (quality < LOW_QUALITY_REPORT_THRESHOLD)
                        .to(dtype=quality.dtype)
                        .mean()
                        .item()
                    ),
                    "low_quality_threshold": LOW_QUALITY_REPORT_THRESHOLD,
                }
            )
        context["dual_space_stats"] = runtime_stats
    # Scores are intentionally unchanged; Dual-Space only modifies geometry
    # and RPR appends the source remote score for newly rescued candidates.
    return output_boxes, pred_score


def validate_dual_space_checkpoint_keys(model, checkpoint_keys):
    """Enforce feature-aware initialization before HEAL's non-strict load."""
    if not getattr(model, "dual_space_enabled", False):
        return
    checkpoint_keys = set(checkpoint_keys)
    model_keys = set(model.state_dict().keys())
    expected_dual = {key for key in model_keys if key.startswith("dual_space_")}
    supplied_dual = {key for key in checkpoint_keys if key.startswith("dual_space_")}
    unexpected_dual = supplied_dual - expected_dual
    mode = model.dual_space_config["mode"]

    if mode == "stage1_anchor" and model.dual_space_config[
        "allow_untrained_initialization"
    ]:
        # Stage1 may start from plain HEAL or a complete lower-profile state.
        # Within every supplied feature group, however, partial state is an
        # error.  The core group is mandatory as soon as any DS key appears.
        required = set()
        if supplied_dual:
            required.update(
                key for key in expected_dual
                if key.startswith(CORE_SHARED_DUAL_SPACE_PREFIXES)
            )
            required.update(
                key for key in expected_dual
                if key.startswith("dual_space_object_adapter_")
            )
            _require_complete_optional_checkpoint_group(
                expected_dual,
                supplied_dual,
                MULTISCALE_SHARED_DUAL_SPACE_PREFIXES
                + ("dual_space_context_adapter_",),
                required,
            )
            _require_complete_optional_checkpoint_group(
                expected_dual,
                supplied_dual,
                QUALITY_SHARED_DUAL_SPACE_PREFIXES,
                required,
            )
    elif mode == "stage2_adapt":
        required = {
            key for key in expected_dual if key.startswith(SHARED_DUAL_SPACE_PREFIXES)
        }
    else:
        required = expected_dual
    missing = required - supplied_dual
    if missing or unexpected_dual:
        details = []
        if missing:
            details.append("missing trained dual-space keys: %s" % ", ".join(sorted(missing)))
        if unexpected_dual:
            details.append("unexpected dual-space keys: %s" % ", ".join(sorted(unexpected_dual)))
        raise RuntimeError("DualSpace checkpoint validation failed; " + "; ".join(details))
    model._dual_space_checkpoint_ready = True
    if mode == "stage1_anchor":
        initialization = (
            "dual_space_resume" if supplied_dual else "base_heal_warm_start"
        )
    elif mode == "stage2_adapt":
        initialization = "stage1_dual_space"
    else:
        initialization = "merged_dual_space"
    print(
        "[DualSpace] checkpoint policy validated | mode=%s | supplied_keys=%d "
        "| initialization=%s"
        % (mode, len(supplied_dual), initialization)
    )


def dual_space_requires_checkpoint(model):
    """Return whether fresh training is forbidden for the configured mode."""
    if not getattr(model, "dual_space_enabled", False):
        return False
    return not bool(getattr(model, "_dual_space_checkpoint_ready", False))


def assert_dual_space_runtime_ready(model):
    """Prevent silent Stage2/inference use of untrained object-space weights."""
    if not getattr(model, "_dual_space_checkpoint_ready", False):
        raise RuntimeError(
            "DualSpace %s weights are not initialized from the required trained "
            "checkpoint; load and validate the required Dual-Space checkpoint before forward"
            % model.dual_space_config["mode"]
        )


def infer_common_bev_channels(args, modality_names):
    """Derive the real post-aligner Common-BEV channel interface from YAML."""
    if not modality_names:
        raise ValueError("dual-space model must define at least one modality")
    channels = []
    for modality in modality_names:
        backbone = args[modality]["backbone_args"]
        upsample = backbone.get("num_upsample_filter", [])
        if upsample:
            value = sum(int(item) for item in upsample)
        else:
            filters = backbone.get("num_filters")
            if not filters:
                raise ValueError("%s backbone_args has no output channel contract" % modality)
            value = int(filters[-1])
        channels.append((modality, value))
    unique = {value for _, value in channels}
    if len(unique) != 1:
        raise ValueError("dual-space Common-BEV channels disagree: %r" % channels)
    return channels[0][1]


def infer_context_bev_channels(args):
    """Return the channel contract of pre-fusion pyramid level 1."""
    fusion = args.get("fusion_backbone")
    if not isinstance(fusion, dict):
        raise TypeError("multi-scale Dual-Space requires fusion_backbone mapping")
    filters = fusion.get("num_filters")
    if not isinstance(filters, (list, tuple)) or len(filters) < 2:
        raise ValueError(
            "multi-scale Dual-Space requires fusion_backbone.num_filters level 1"
        )
    value = filters[1]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("fusion_backbone.num_filters[1] must be positive integer")
    return value


def _warp_scene_features(features, transforms, mode):
    agent_count, channels, height, width = features.shape
    grid = F.affine_grid(
        transforms,
        (agent_count, channels, height, width),
        align_corners=False,
    )
    return F.grid_sample(
        features,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def _agent_positions_in_ego(ego_to_agent, reference):
    if not torch.is_tensor(ego_to_agent) or ego_to_agent.ndim != 3:
        raise ValueError("ego-to-agent transforms must have shape [A,4,4]")
    if tuple(ego_to_agent.shape[1:]) != (4, 4):
        raise ValueError("ego-to-agent transforms must have shape [A,4,4]")
    transforms = ego_to_agent.to(device=reference.device, dtype=reference.dtype)
    # pairwise[ego, agent] maps ego coordinates to agent coordinates, exactly
    # the output-to-input direction used by affine_grid.  Its inverse maps the
    # agent origin into ego coordinates.
    agent_to_ego = torch.linalg.inv(transforms)
    return agent_to_ego[:, :2, 3]


def _scene_anchor_for_decoder(anchor_box, scene_index, scene_count):
    """Select one anchor grid without inventing an agent/batch dimension.

    HEAL's real collate path shares one unbatched ``[H,W,A,7]`` anchor grid
    across the batch.  A synthetic or alternate loader may provide
    ``[B,H,W,A,7]`` instead.  ``VoxelPostprocessor.delta_to_boxes3d`` already
    repeats the selected grid for the dense prediction batch, so repeating it
    here would corrupt the anchor count for per-agent RPR decoding.
    """
    if (
        not torch.is_tensor(anchor_box)
        or anchor_box.ndim < 1
        or anchor_box.shape[-1] != 7
    ):
        raise ValueError("anchor_box must be a tensor ending in dimension 7")
    if anchor_box.ndim == 4:
        return anchor_box
    if anchor_box.ndim == 5:
        if int(anchor_box.shape[0]) != int(scene_count):
            raise ValueError("batched anchor_box does not match scene count")
        if not 0 <= int(scene_index) < int(scene_count):
            raise ValueError("scene_index is outside the anchor_box batch")
        return anchor_box[scene_index]
    raise ValueError(
        "anchor_box must have shape [H,W,A,7] or [B,H,W,A,7]"
    )


def _quality_scalar(value, reference, name):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError("%s must have shape [M]" % name)
    if value.shape[0] != reference.shape[0]:
        raise ValueError("%s count must match object embeddings" % name)
    return value.to(device=reference.device, dtype=reference.dtype).unsqueeze(-1)


def _require_complete_optional_checkpoint_group(
    expected, supplied, prefixes, required
):
    group_expected = {key for key in expected if key.startswith(prefixes)}
    group_supplied = group_expected & supplied
    if group_supplied:
        required.update(group_expected)


def _build_source_support(model, features, modalities):
    support = features.new_ones((features.shape[0], 1, features.shape[2], features.shape[3]))
    height, width = int(features.shape[2]), int(features.shape[3])
    edge_margin = 4 if not model.training else 0
    for index, modality in enumerate(modalities):
        if model.sensor_type_dict.get(modality) != "camera":
            continue
        ratio_h = float(getattr(model, "crop_ratio_H_%s" % modality))
        ratio_w = float(getattr(model, "crop_ratio_W_%s" % modality))
        valid_h = min(height, max(0, int(round(height / ratio_h)) - edge_margin))
        valid_w = min(width, max(0, int(round(width / ratio_w)) - edge_margin))
        start_h = (height - valid_h) // 2
        start_w = (width - valid_w) // 2
        support[index].zero_()
        support[index, :, start_h:start_h + valid_h, start_w:start_w + valid_w] = 1
    return support


def _set_module_trainability(module, trainable, parent_training):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
    module.train(bool(trainable and parent_training))


def _print_dual_space_summary(model, shared_trainable, active_modality):
    config = model.dual_space_config
    named_parameters = tuple(model.named_parameters())
    object_parameters = [
        parameter for name, parameter in named_parameters
        if name.startswith("dual_space_")
    ]
    core_parameters = [
        parameter for name, parameter in named_parameters
        if name.startswith(CORE_SHARED_DUAL_SPACE_PREFIXES)
    ]
    multi_parameters = [
        parameter for name, parameter in named_parameters
        if name.startswith(MULTISCALE_SHARED_DUAL_SPACE_PREFIXES)
        or name.startswith("dual_space_context_adapter_")
    ]
    quality_parameters = [
        parameter for name, parameter in named_parameters
        if name.startswith(QUALITY_SHARED_DUAL_SPACE_PREFIXES)
    ]
    active_adapter_parameters = []
    if active_modality is not None:
        active_prefixes = (
            "dual_space_object_adapter_%s." % active_modality,
            "dual_space_context_adapter_%s." % active_modality,
        )
        active_adapter_parameters = [
            parameter for name, parameter in named_parameters
            if name.startswith(active_prefixes)
        ]
    total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
    all_trainable = sum(
        parameter.numel() for _, parameter in named_parameters
        if parameter.requires_grad
    )
    object_trainable = sum(
        parameter.numel() for parameter in object_parameters if parameter.requires_grad
    )
    print("[DualSpace]")
    print("profile=%s" % config.get("experiment_profile", config["version"]))
    print("enabled=True")
    print("version=%s" % config["version"])
    print("mode=%s" % config["mode"])
    print("roi_size=%s" % config["roi"]["output_size"])
    print("embedding_dim=%s" % config["object_encoder"]["embedding_dim"])
    print("consensus=%s" % config["consensus"]["mode"])
    print("multi_scale=%s" % model.dual_space_flags["multi_scale"])
    if model.dual_space_flags["multi_scale"]:
        print("fusion=%s" % config["multi_scale"]["fusion"])
        print("detail_roi_size=%s" % (model.dual_space_object_roi.output_size,))
        print("context_roi_size=%s" % (model.dual_space_context_roi.output_size,))
    print("quality=%s" % model.dual_space_flags["quality"])
    print("RPR=%s" % model.dual_space_flags["remote_proposal_rescue"])
    print("proposal_source=%s" % config["training_proposals"]["source"])
    print("shared_object_trainable=%s" % shared_trainable)
    if active_modality is not None:
        print("active_modality=%s" % active_modality)
        print("adapter_%s_trainable=True" % active_modality)
    print("base parameters=%d" % (total_parameters - sum(
        parameter.numel() for parameter in object_parameters
    )))
    print("dual-space core parameters=%d" % sum(
        parameter.numel() for parameter in core_parameters
    ))
    print("multi-scale parameters=%d" % sum(
        parameter.numel() for parameter in multi_parameters
    ))
    print("quality parameters=%d" % sum(
        parameter.numel() for parameter in quality_parameters
    ))
    print("active modality adapter parameters=%d" % sum(
        parameter.numel() for parameter in active_adapter_parameters
    ))
    print("object trainable params=%d" % object_trainable)
    print("base trainable params=%d" % (all_trainable - object_trainable))
    print("total trainable params=%d" % all_trainable)
    print("total frozen params=%d" % (total_parameters - all_trainable))


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
