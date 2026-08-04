"""Modality-local low-rank common projectors and shared decoders."""

import torch
import torch.nn as nn


def _activation(name):
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError("Unsupported Open-DCSI projector activation: {}".format(name))


def _normalization(name, channels):
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "batchnorm":
        return nn.BatchNorm2d(channels)
    if name == "groupnorm":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("Unsupported Open-DCSI projector norm: {}".format(name))


class CommonProjector(nn.Module):
    """Project an aligned HEAL scale into a compact modality-neutral space."""

    def __init__(self, input_channels, common_channels, config):
        super().__init__()
        projector_type = config["type"].lower()
        if projector_type != "pointwise_depthwise":
            raise ValueError(
                "Unsupported Open-DCSI projector type: {}".format(projector_type)
            )
        self.pointwise = nn.Conv2d(input_channels, common_channels, 1, bias=True)
        self.depthwise = nn.Conv2d(
            common_channels,
            common_channels,
            kernel_size=3,
            padding=1,
            groups=common_channels,
            bias=True,
        )
        self.norm = _normalization(config["norm"], common_channels)
        self.activation = _activation(config["activation"])
        self.use_residual = bool(config["residual"])
        self.residual_projection = None
        if self.use_residual and input_channels != common_channels:
            self.residual_projection = nn.Conv2d(
                input_channels, common_channels, kernel_size=1, bias=False
            )

    def forward(self, feature):
        common = self.pointwise(feature)
        common = self.depthwise(common)
        common = self.norm(common)
        if self.use_residual:
            residual = feature
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            common = common + residual
        return self.activation(common)


class CommonDecoder(nn.Module):
    """Decode one compact common scale to the official HEAL scale channels."""

    def __init__(self, common_channels, output_channels, zero_init_residual=True):
        super().__init__()
        self.base = nn.Conv2d(common_channels, output_channels, 1, bias=True)
        self.residual = None
        if zero_init_residual:
            self.residual = nn.Conv2d(common_channels, output_channels, 1, bias=True)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, common):
        decoded = self.base(common)
        if self.residual is not None:
            decoded = decoded + self.residual(common)
        return decoded


def split_project_by_modality(feature, agent_modalities, projectors):
    """Apply local projectors without assuming modality order or agent count."""

    if feature.shape[0] != len(agent_modalities):
        raise ValueError("Open-DCSI agent modality count does not match feature batch")
    common = [None] * len(agent_modalities)
    modality_indices = {}
    for index, modality in enumerate(agent_modalities):
        modality_indices.setdefault(modality, []).append(index)
    for modality, indices in modality_indices.items():
        if modality not in projectors:
            raise KeyError(
                "Open-DCSI has no common projector for modality {}".format(modality)
            )
        index_tensor = torch.as_tensor(indices, device=feature.device, dtype=torch.long)
        projected = projectors[modality](feature.index_select(0, index_tensor))
        for local_index, agent_index in enumerate(indices):
            common[agent_index] = projected[local_index]
    return torch.stack(common, dim=0)
