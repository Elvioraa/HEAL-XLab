"""Independent Stage2 HEAL wrapper for incremental Open-DCSI development."""

from opencood.models.heter_pyramid_single import HeterPyramidSingle
from opencood.models.sub_modules.open_dcsi.model_bridge import (
    forward_with_open_dcsi,
    initialize_open_dcsi,
)
from opencood.models.sub_modules.open_dcsi.stage2 import (
    configure_stage2_independent,
    enforce_stage2_shared_eval,
)


class HeterPyramidSingleOpenDcsiStage2(HeterPyramidSingle):
    """Preserve official Stage2 freezing and forward behavior when disabled."""

    def __init__(self, args):
        super().__init__(args)
        initialize_open_dcsi(self, args, bridge_single=True)
        configure_stage2_independent(self)

    def parameters(self, recurse=True):
        parameters = super().parameters(recurse=recurse)
        if not getattr(self, "_open_dcsi_filter_optimizer_parameters", False):
            yield from parameters
            return
        for parameter in parameters:
            if parameter.requires_grad:
                yield parameter

    def train(self, mode=True):
        result = super().train(mode)
        if mode and getattr(self, "open_dcsi_enabled", False):
            for module_name in self.fix_modules:
                getattr(self, module_name).eval()
            enforce_stage2_shared_eval(self)
        return result

    def forward(self, data_dict):
        if not self.open_dcsi_enabled:
            return super().forward(data_dict)
        return forward_with_open_dcsi(
            self, super().forward, data_dict, single=True, stage="stage2"
        )
