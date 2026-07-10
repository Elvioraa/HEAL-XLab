"""Local evidence head for PACT-CBEA v1 Feature Mode."""

from opencood.models.hvp_heal_v3.evidence_head import HvpHealV3EvidenceHead


class PACTCBEALocalEvidenceHead(HvpHealV3EvidenceHead):
    """PACT-named wrapper around the stable HVP-v3 evidence head.

    Descriptor output is kept available for checkpoint compatibility, but PACT
    v1 training configs disable descriptor loss and descriptor consistency.
    """

    pass
