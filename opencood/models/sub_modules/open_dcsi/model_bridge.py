"""Opt-in bridge from official PyramidFusion calls to Open-DCSI modules."""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse
from opencood.models.sub_modules.open_dcsi.common_fusion import (
    LowRankCommonFusion,
)
from opencood.models.sub_modules.open_dcsi.common_projector import (
    CommonDecoder,
    CommonProjector,
    split_project_by_modality,
)
from opencood.models.sub_modules.open_dcsi.innovation_aggregator import (
    InnovationAggregator,
)
from opencood.models.sub_modules.open_dcsi.cross_scale_geometry import (
    CrossScaleGeometrySampler,
)
from opencood.models.sub_modules.open_dcsi.geometry_refiner import GeometryRefiner
from opencood.models.sub_modules.open_dcsi.innovation_tokenizer import (
    InnovationTokenizer,
    transform_tokens_to_ego,
)
from opencood.models.sub_modules.open_dcsi.quality_router import (
    EvidenceQualityRouter,
)
from opencood.models.sub_modules.open_dcsi.packet_codec import CommunicationManager
from opencood.models.sub_modules.open_dcsi.streaming_fusion import (
    StreamingCommonFusion,
)
from opencood.models.sub_modules.open_dcsi.dense_innovation_fusion import (
    DenseInnovationFusion,
)
from opencood.models.sub_modules.open_dcsi.config import (
    is_open_dcsi_enabled,
    validate_open_dcsi_config,
)


PHASE2_IMPLEMENTED_MODULES = {
    "common_space",
    "common_space.projector",
    "common_space.decoder",
    "common_space.reconstruction",
    "common_space.common_detection_supervision",
    "common_fusion",
    "common_fusion.absolute_reject",
    "diagnostics",
}

PHASE3_IMPLEMENTED_MODULES = PHASE2_IMPLEMENTED_MODULES | {
    "innovation_tokens",
    "innovation_quality",
    "innovation_quality.calibration",
    "innovation_aggregation",
    "innovation_aggregation.geometric_clustering",
    "innovation_aggregation.absolute_reject",
}

PHASE4_IMPLEMENTED_MODULES = PHASE3_IMPLEMENTED_MODULES | {
    "cross_scale_geometry",
    "geometry_refiner",
}

PHASE5_IMPLEMENTED_MODULES = PHASE4_IMPLEMENTED_MODULES | {
    "open_heterogeneous",
    "stage2_independent",
}

PHASE6_IMPLEMENTED_MODULES = PHASE5_IMPLEMENTED_MODULES | {
    "communication",
    "communication.common_codec",
    "communication.token_codec",
    "communication.budget",
    "communication.selection",
}

PHASE7_IMPLEMENTED_MODULES = PHASE6_IMPLEMENTED_MODULES | {
    "streaming_fusion",
}

PHASE8_IMPLEMENTED_MODULES = PHASE7_IMPLEMENTED_MODULES | {
    "losses",
    "losses.common_detection",
    "losses.reconstruction",
    "losses.common_innovation_decorrelation",
    "losses.innovation_detection",
    "losses.box_refinement",
    "losses.quality",
    "losses.token_sparsity",
    "losses.budget",
}

PHASE9_IMPLEMENTED_MODULES = PHASE8_IMPLEMENTED_MODULES | {
    "dense_innovation_map",
}


class OpenDCSICommonRuntime(nn.Module):
    """Own common-space parameters and execute bridged pyramid operations."""

    def __init__(self, args, config):
        super().__init__()
        self.config = config
        self.modality_names = [
            key for key in args if key.startswith("m") and key[1:].isdigit()
        ]
        scale_channels = list(args["fusion_backbone"]["num_filters"])
        self.scale_channels = scale_channels
        self.lidar_range = tuple(float(value) for value in args["lidar_range"])
        ratio = float(config["common_space"]["channel_ratio"])
        minimum = int(config["common_space"]["min_channels"])
        self.common_channels = [
            min(channels, max(minimum, int(round(channels * ratio))))
            for channels in scale_channels
        ]

        projector_config = config["common_space"]["projector"]
        self.common_projectors = nn.ModuleDict()
        for modality in self.modality_names:
            self.common_projectors[modality] = nn.ModuleList(
                [
                    CommonProjector(channels, common_channels, projector_config)
                    for channels, common_channels in zip(
                        scale_channels, self.common_channels
                    )
                ]
            )
        decoder_config = config["common_space"]["decoder"]
        self.common_decoders = nn.ModuleList(
            [
                CommonDecoder(
                    common_channels,
                    channels,
                    decoder_config["zero_init_residual"],
                )
                for channels, common_channels in zip(
                    scale_channels, self.common_channels
                )
            ]
        )
        if config["common_fusion"]["enabled"]:
            self.common_fusions = nn.ModuleList(
                [
                    LowRankCommonFusion(config["common_fusion"])
                    for _ in scale_channels
                ]
            )
            gate_count = (
                len(scale_channels)
                if config["common_fusion"]["scale_specific_gate"]
                else 1
            )
            self.common_residual_gate = nn.Parameter(torch.zeros(gate_count))
        if config["innovation_tokens"]["enabled"]:
            self.innovation_tokenizer = InnovationTokenizer(
                self.modality_names,
                scale_channels,
                self.lidar_range,
                config["innovation_tokens"],
                config["innovation_quality"],
            )
            self.innovation_quality_router = EvidenceQualityRouter(
                config["innovation_quality"]
            )
            if config["innovation_aggregation"]["enabled"]:
                self.innovation_aggregator = InnovationAggregator(
                    config["innovation_aggregation"]
                )
            if config["cross_scale_geometry"]["enabled"]:
                self.cross_scale_geometry = CrossScaleGeometrySampler(
                    scale_channels,
                    int(config["innovation_tokens"]["token_dim"]),
                    int(config["innovation_tokens"]["geometry_dim"]),
                    self.lidar_range,
                    config["cross_scale_geometry"],
                )
            if config["geometry_refiner"]["enabled"]:
                self.geometry_refiner = GeometryRefiner(
                    int(config["innovation_tokens"]["token_dim"]),
                    int(config["innovation_tokens"]["geometry_dim"]),
                    config["geometry_refiner"],
                )
        if config["dense_innovation_map"]["enabled"]:
            self.dense_innovation_fusion = DenseInnovationFusion(
                config["dense_innovation_map"]
            )
        if config["communication"]["enabled"] or config["streaming_fusion"][
            "enabled"
        ]:
            self.communication = CommunicationManager(config["communication"])
        if config["streaming_fusion"]["enabled"]:
            self.streaming_common_fusion = StreamingCommonFusion(
                config["common_fusion"], config["streaming_fusion"]
            )
        self._runtime_output = None

    def _gate_for_scale(self, scale_index):
        if self.common_residual_gate.numel() == 1:
            return self.common_residual_gate[0]
        return self.common_residual_gate[scale_index]

    def _project(self, feature, agent_modalities, scale_index):
        projectors = {
            modality: modules[scale_index]
            for modality, modules in self.common_projectors.items()
        }
        return split_project_by_modality(feature, agent_modalities, projectors)

    def _set_runtime_output(self, **items):
        self._runtime_output = items

    def consume_runtime_output(self):
        output = self._runtime_output or {}
        self._runtime_output = None
        return output

    def _tokenize(self, innovation_features, occ_outputs, modalities, record_len, align_corners):
        if not hasattr(self, "innovation_tokenizer"):
            return None
        return self.innovation_tokenizer(
            innovation_features,
            occ_outputs,
            modalities,
            record_len,
            align_corners=align_corners,
        )

    def _trim_runtime_diagnostics(self):
        keep_for_loss = self.training and self.config["losses"]["enabled"]
        if self.config["diagnostics"]["enabled"] or keep_for_loss:
            return
        for key in (
            "common_features",
            "reconstructed_features",
            "innovation_features",
            "fused_common_features",
            "common_fusion_weights",
            "fused_dense_innovation_features",
        ):
            self._runtime_output.pop(key, None)

    def finalize_tokens(self, data_dict, single=False):
        if self._runtime_output is None:
            return
        local_tokens = self._runtime_output.get("innovation_tokens_local")
        if local_tokens is None:
            self._runtime_output.pop("innovation_tokens_local", None)
            self._runtime_output.pop("_record_len", None)
            self._runtime_output.pop("_affine_matrix", None)
            self._runtime_output.pop("_align_corners", None)
            self._trim_runtime_diagnostics()
            return
        if single:
            tokens_ego = dict(local_tokens)
            tokens_ego["centers_ego"] = local_tokens["centers_local"]
            tokens_ego["yaw_sin_cos_ego"] = local_tokens["yaw_sin_cos_local"]
            tokens_ego["boxes_ego_hwl"] = local_tokens["boxes_local_hwl"]
            tokens_ego["coordinate_frame"] = "ego_metric"
        else:
            pairwise = data_dict.get("pairwise_t_matrix")
            if pairwise is None:
                raise ValueError("Open-DCSI token transform requires pairwise_t_matrix")
            tokens_ego = transform_tokens_to_ego(
                local_tokens, pairwise, data_dict["record_len"]
            )
        self._runtime_output["innovation_tokens"] = tokens_ego
        self._runtime_output.pop("innovation_tokens_local", None)
        if hasattr(self, "innovation_aggregator"):
            fused = self.innovation_aggregator(
                tokens_ego, self.innovation_quality_router
            )
        else:
            fused = tokens_ego
        self._runtime_output["fused_tokens"] = fused
        self._runtime_output["quality"] = {
            "token_reliability": self.innovation_quality_router(tokens_ego)
        }
        geometry_output = None
        if hasattr(self, "cross_scale_geometry"):
            geometry_output = self.cross_scale_geometry(
                fused,
                self._runtime_output["innovation_features"],
                self._runtime_output["_record_len"],
                self._runtime_output["_affine_matrix"],
                self._runtime_output["_align_corners"],
            )
            self._runtime_output["cross_scale_geometry"] = geometry_output
        if hasattr(self, "geometry_refiner"):
            context = geometry_output["context"] if geometry_output is not None else None
            self._runtime_output["geometry_refinement"] = self.geometry_refiner(
                fused, context
            )
        self._runtime_output.pop("_record_len", None)
        self._runtime_output.pop("_affine_matrix", None)
        self._runtime_output.pop("_align_corners", None)
        self._trim_runtime_diagnostics()

    def forward_collab(
        self,
        pyramid,
        spatial_features,
        record_len,
        affine_matrix,
        agent_modality_list=None,
        cam_crop_info=None,
        **kwargs
    ):
        if agent_modality_list is None:
            raise ValueError("Open-DCSI collaborative fusion requires agent modalities")
        feature_list = pyramid.get_multiscale_feature(spatial_features)
        occ_outputs = []
        score_outputs = []
        common_features = []
        reconstructed_features = []
        innovation_features = []
        fused_common_features = []
        fusion_weights = []
        decoded_fused_features = []
        official_fused_features = []
        quality_prior = kwargs.get("cbea_alpha")

        for scale_index, feature in enumerate(feature_list):
            occ_map = getattr(pyramid, "single_head_{}".format(scale_index))(feature)
            occ_outputs.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4
            score_outputs.append(score)
            common = self._project(feature, agent_modality_list, scale_index)
            reconstructed = self.common_decoders[scale_index](common)
            innovation = feature - reconstructed
            common_features.append(common)
            reconstructed_features.append(reconstructed)
            innovation_features.append(innovation)

        communication_state = None
        communication_stats = None
        fusion_common_features = common_features
        communication_masks = [torch.ones_like(score) for score in score_outputs]
        use_streaming = hasattr(self, "streaming_common_fusion") and not self.training
        if hasattr(self, "communication"):
            (
                fusion_common_features,
                communication_masks,
                communication_state,
            ) = self.communication.process_common(
                common_features,
                score_outputs,
                record_len,
                materialize_dense=not use_streaming,
            )

        streaming_scale_stats = []
        for scale_index, (feature, score, common) in enumerate(
            zip(feature_list, score_outputs, fusion_common_features)
        ):
            prior = communication_masks[scale_index]
            if quality_prior is not None:
                resized_prior = quality_prior
                if resized_prior.shape[-2:] != score.shape[-2:]:
                    resized_prior = F.interpolate(
                        resized_prior,
                        size=score.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                prior = prior * resized_prior.to(device=prior.device, dtype=prior.dtype)
            if use_streaming:
                fused_common, scale_streaming_stats = self.streaming_common_fusion(
                    common,
                    score,
                    record_len,
                    affine_matrix,
                    self._gate_for_scale(scale_index),
                    communication_state,
                    scale_index,
                    quality_prior=prior,
                    align_corners=pyramid.align_corners,
                )
                weights = []
                streaming_scale_stats.append(scale_streaming_stats)
            elif self.config["common_fusion"]["enabled"]:
                fusion = self.common_fusions[scale_index]
                fusion.align_corners = pyramid.align_corners
                fused_common, weights = fusion(
                    common,
                    score,
                    record_len,
                    affine_matrix,
                    self._gate_for_scale(scale_index),
                    quality_prior=prior,
                )
            else:
                fused_common = weighted_fuse(
                    common,
                    score * communication_masks[scale_index],
                    record_len,
                    affine_matrix,
                    pyramid.align_corners,
                )
                weights = []

            decoded_fused = self.common_decoders[scale_index](fused_common)
            if not self.config["common_space"]["common_detection_supervision"][
                "enabled"
            ]:
                official_fused_features.append(
                    weighted_fuse(
                        feature,
                        score,
                        record_len,
                        affine_matrix,
                        pyramid.align_corners,
                        cbea_alpha=kwargs.get("cbea_alpha"),
                        cbea_lambda=kwargs.get("cbea_lambda", 0.0),
                        cbea_exclude_threshold=kwargs.get(
                            "cbea_exclude_threshold", 0.0
                        ),
                        cbea_exclude_floor_mix=kwargs.get(
                            "cbea_exclude_floor_mix", 0.0
                        ),
                        cbea_injection_strength=kwargs.get(
                            "cbea_injection_strength", 0.0
                        ),
                    )
                )
            fused_common_features.append(fused_common)
            fusion_weights.append(weights)
            decoded_fused_features.append(decoded_fused)

        detection_features = (
            decoded_fused_features
            if self.config["common_space"]["common_detection_supervision"][
                "enabled"
            ]
            else official_fused_features
        )
        fused_dense_innovation_features = None
        if hasattr(self, "dense_innovation_fusion"):
            fused_dense_innovation_features = self.dense_innovation_fusion(
                innovation_features,
                score_outputs,
                record_len,
                affine_matrix,
                pyramid.align_corners,
                cbea_alpha=kwargs.get("cbea_alpha"),
                cbea_lambda=kwargs.get("cbea_lambda", 0.0),
                cbea_exclude_threshold=kwargs.get("cbea_exclude_threshold", 0.0),
                cbea_exclude_floor_mix=kwargs.get("cbea_exclude_floor_mix", 0.0),
                cbea_injection_strength=kwargs.get("cbea_injection_strength", 0.0),
            )
            detection_features = self.dense_innovation_fusion.add_to_detection_features(
                detection_features, fused_dense_innovation_features
            )
        final_feature = pyramid.decode_multiscale_feature(detection_features)
        local_tokens = self._tokenize(
            innovation_features,
            occ_outputs,
            agent_modality_list,
            record_len,
            pyramid.align_corners,
        )
        if (
            local_tokens is not None
            and hasattr(self, "communication")
            and communication_state is not None
        ):
            local_tokens, communication_stats = self.communication.process_tokens(
                local_tokens,
                self.innovation_quality_router,
                communication_state,
            )
        elif communication_state is not None:
            self.communication._finish_stats(communication_state)
            communication_stats = communication_state["stats"]
        streaming_stats = None
        if streaming_scale_stats:
            streaming_stats = {
                "processed_common_packets": sum(
                    item["processed_common_packets"] for item in streaming_scale_stats
                ),
                "fusion_local_peak_bytes": max(
                    item["fusion_local_peak_bytes"] for item in streaming_scale_stats
                ),
                "recovered_full_dense_stack": False,
                "scale_stats": streaming_scale_stats,
            }
        self._set_runtime_output(
            common_features=common_features,
            reconstructed_features=reconstructed_features,
            innovation_features=innovation_features,
            fused_dense_innovation_features=fused_dense_innovation_features,
            fused_common_features=fused_common_features,
            common_fusion_weights=fusion_weights,
            innovation_tokens_local=local_tokens,
            _record_len=record_len,
            _affine_matrix=affine_matrix,
            _align_corners=pyramid.align_corners,
            communication_stats=communication_stats,
            streaming_stats=streaming_stats,
        )
        return final_feature, occ_outputs

    def forward_single(self, pyramid, spatial_features, modality_names):
        feature_list = pyramid.get_multiscale_feature(spatial_features)
        occ_outputs = []
        score_outputs = []
        common_features = []
        reconstructed_features = []
        innovation_features = []
        for scale_index, feature in enumerate(feature_list):
            occ_outputs.append(
                getattr(pyramid, "single_head_{}".format(scale_index))(feature)
            )
            score_outputs.append(torch.sigmoid(occ_outputs[-1]) + 1e-4)
            if isinstance(modality_names, str):
                modalities = [modality_names] * int(feature.shape[0])
            else:
                modalities = list(modality_names)
            if len(modalities) != int(feature.shape[0]):
                raise ValueError(
                    "Open-DCSI single feature count does not match agent modalities"
                )
            common = self._project(feature, modalities, scale_index)
            reconstructed = self.common_decoders[scale_index](common)
            common_features.append(common)
            reconstructed_features.append(reconstructed)
            innovation_features.append(feature - reconstructed)
        detection_features = (
            reconstructed_features
            if self.config["common_space"]["common_detection_supervision"][
                "enabled"
            ]
            else feature_list
        )
        fused_dense_innovation_features = None
        if hasattr(self, "dense_innovation_fusion"):
            fused_dense_innovation_features = innovation_features
            detection_features = self.dense_innovation_fusion.add_to_detection_features(
                detection_features, fused_dense_innovation_features
            )
        final_feature = pyramid.decode_multiscale_feature(detection_features)
        record_len = torch.ones(
            spatial_features.shape[0],
            device=spatial_features.device,
            dtype=torch.long,
        )
        local_tokens = self._tokenize(
            innovation_features,
            occ_outputs,
            modalities,
            record_len,
            pyramid.align_corners,
        )
        communication_stats = None
        if hasattr(self, "communication"):
            _, _, communication_state = self.communication.process_common(
                common_features, score_outputs, record_len
            )
            if local_tokens is not None:
                local_tokens, communication_stats = self.communication.process_tokens(
                    local_tokens,
                    self.innovation_quality_router,
                    communication_state,
                )
        identity_affine = spatial_features.new_zeros(
            (spatial_features.shape[0], 1, 1, 2, 3)
        )
        identity_affine[..., 0, 0] = 1.0
        identity_affine[..., 1, 1] = 1.0
        self._set_runtime_output(
            common_features=common_features,
            reconstructed_features=reconstructed_features,
            innovation_features=innovation_features,
            fused_dense_innovation_features=fused_dense_innovation_features,
            fused_common_features=common_features,
            common_fusion_weights=[],
            innovation_tokens_local=local_tokens,
            _record_len=record_len,
            _affine_matrix=identity_affine,
            _align_corners=pyramid.align_corners,
            communication_stats=communication_stats,
        )
        return final_feature, occ_outputs


def initialize_open_dcsi(owner, args, bridge_collab=False, bridge_single=False):
    """Validate config and install bridge methods only for an enabled feature."""

    raw_config = args.get("open_dcsi")
    owner.open_dcsi_config = validate_open_dcsi_config(
        raw_config, implemented_modules=PHASE9_IMPLEMENTED_MODULES
    )
    owner.open_dcsi_enabled = is_open_dcsi_enabled(raw_config)
    if not owner.open_dcsi_enabled:
        return
    print(
        "Open-DCSI normalized config: {}".format(
            json.dumps(owner.open_dcsi_config, sort_keys=True)
        )
    )
    if not owner.open_dcsi_config["common_space"]["enabled"]:
        return
    owner.open_dcsi = OpenDCSICommonRuntime(args, owner.open_dcsi_config)
    if bridge_collab:
        def bridged_collab(
            spatial_features,
            record_len,
            affine_matrix,
            agent_modality_list=None,
            cam_crop_info=None,
            **kwargs
        ):
            return owner.open_dcsi.forward_collab(
                owner.pyramid_backbone,
                spatial_features,
                record_len,
                affine_matrix,
                agent_modality_list,
                cam_crop_info,
                **kwargs
            )

        owner.pyramid_backbone.forward_collab = bridged_collab
    if bridge_single:
        def bridged_single(spatial_features):
            modality_names = owner._open_dcsi_single_modalities
            return owner.open_dcsi.forward_single(
                owner.pyramid_backbone, spatial_features, modality_names
            )

        owner.pyramid_backbone.forward_single = bridged_single


def _audit_open_heterogeneous_forward(owner, data_dict, stage):
    if not owner.open_dcsi_config["open_heterogeneous"]["enabled"]:
        return
    if stage == "stage1":
        modalities = data_dict.get("agent_modality_list", [])
        if not modalities or any(modality != "m1" for modality in modalities):
            raise ValueError("Open-DCSI Stage1 requires homogeneous m1 agents")
    elif stage == "stage2":
        from opencood.models.sub_modules.open_dcsi.stage2 import audit_stage2_batch

        audit_stage2_batch(owner, data_dict)


def forward_with_open_dcsi(owner, parent_forward, data_dict, single=False, stage="inference"):
    """Run an enabled bridge and attach only outputs required by Open-DCSI."""

    if not owner.open_dcsi_enabled or not hasattr(owner, "open_dcsi"):
        return parent_forward(data_dict)
    _audit_open_heterogeneous_forward(owner, data_dict, stage)
    if single:
        input_keys = [key for key in data_dict if key.startswith("inputs_")]
        if len(input_keys) != 1:
            raise ValueError("Open-DCSI Stage2 requires exactly one input modality")
        owner._open_dcsi_single_modalities = input_keys[0][len("inputs_"):]
    else:
        owner._open_dcsi_single_modalities = data_dict.get("agent_modality_list", [])
    output = parent_forward(data_dict)
    owner.open_dcsi.finalize_tokens(data_dict, single=single)
    open_output = owner.open_dcsi.consume_runtime_output()
    if "anchor_box" in data_dict:
        open_output["anchor_box"] = data_dict["anchor_box"]
    if owner.open_dcsi_config["diagnostics"]["enabled"]:
        open_output["config_summary"] = {
            "enabled": True,
            "open_heterogeneous": owner.open_dcsi_config["open_heterogeneous"][
                "enabled"
            ],
            "common_space": True,
            "common_fusion": owner.open_dcsi_config["common_fusion"]["enabled"],
            "common_channels": list(owner.open_dcsi.common_channels),
            "innovation_tokens": owner.open_dcsi_config["innovation_tokens"][
                "enabled"
            ],
            "cross_scale_geometry": owner.open_dcsi_config[
                "cross_scale_geometry"
            ]["enabled"],
            "geometry_refiner": owner.open_dcsi_config["geometry_refiner"][
                "enabled"
            ],
            "dense_innovation_map": owner.open_dcsi_config[
                "dense_innovation_map"
            ]["enabled"],
            "communication": owner.open_dcsi_config["communication"]["enabled"],
            "streaming_fusion": owner.open_dcsi_config["streaming_fusion"][
                "enabled"
            ],
            "losses": owner.open_dcsi_config["losses"]["enabled"],
        }
    output["open_dcsi"] = open_output
    return output
