# HEAL-XLab-v1-HBEC

Full name: Hypothesis-guided Bayesian Evidence Communication.

HBEC moves experimental communication from feature space toward object-state, hypothesis-space, probabilistic evidence communication at the final-infer post-process stage.

Implemented components:

- Hypothesis and evidence packets with boxes, scores, optional labels, uncertainty, source metadata, transform status, and payload estimate.
- BEV center-distance matcher with BEV IoU support through existing HEAL utilities when available.
- Greedy matching with configurable IoU, distance, and combined-score weights.
- Bayesian refiner using inverse-variance weighted box averaging and yaw sin/cos averaging.
- Score fusion through logit evidence accumulation.
- Novel object insertion with configurable score, distance, and max-count thresholds.
- Optional refute/suppress with default no-op suppression factor.
- Payload and debug metrics recorder.

Safety behavior:

- No ground truth is used for fusion.
- No raw camera feature is used or bypassed.
- No training code or model parameters are changed.
- If collaborator evidence is unavailable, HBEC returns the official output unchanged.
- If HBEC raises an error and `fallback_on_error` is true, the official output is returned unchanged.

Current evidence limitation:

The audited intermediate HEAL final-infer path does not expose reliable collaborator-only object predictions. Therefore, enabled HBEC currently needs an explicit `collaborator_evidence` packet in `infer_context`; otherwise it records `fallback_reason = no_collaborator_evidence`.

