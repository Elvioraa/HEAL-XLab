"""Stage1 HEAL wrapper for incremental Open-DCSI development."""

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.open_dcsi.model_bridge import (
    forward_with_open_dcsi,
    initialize_open_dcsi,
)


class HeterPyramidCollabOpenDcsiStage1(HeterPyramidCollab):
    """Preserve the official Stage1 path until Open-DCSI modules are enabled."""

    def __init__(self, args):
        super().__init__(args)
        initialize_open_dcsi(self, args, bridge_collab=True)

    def forward(self, data_dict):
        if not self.open_dcsi_enabled:
            return super().forward(data_dict)
        return forward_with_open_dcsi(
            self, super().forward, data_dict, stage="stage1"
        )
