"""Final heterogeneous PACT-CBEA/HEAL wrapper for Open-DCSI."""

from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.sub_modules.open_dcsi.model_bridge import (
    forward_with_open_dcsi,
    initialize_open_dcsi,
)


class HeterPyramidCollabOpenDcsi(HeterPyramidCollabPactCbea):
    """Extend the real PACT-CBEA -> HEAL collaboration inheritance chain."""

    def __init__(self, args):
        super().__init__(args)
        initialize_open_dcsi(
            self, args, bridge_collab=True, bridge_single=True
        )

    def forward(self, data_dict):
        if not self.open_dcsi_enabled:
            return super().forward(data_dict)
        return forward_with_open_dcsi(self, super().forward, data_dict)
