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

## v1.1 Object-level Evidence Extraction

HEAL-XLab-v1.1 adds `HBECEvidenceExtractor` in `opencood/xlab/hbec/evidence.py`.

Supported `evidence_source` values:

- `none`: default, no extraction, fallback unchanged.
- `explicit`: use an evidence packet supplied by `infer_context`.
- `late_fusion_reinfer`: call official `inference_utils.inference_late_fusion()` and convert returned `pred_box_tensor` / `pred_score` to an object-level `EvidencePacket`.
- `no_fusion_reinfer`: call official `inference_utils.inference_no_fusion()` only if the active dataset supports `post_process_no_fusion`.

The re-inference evidence is object-level prediction output after official post-process, not raw feature bypass. Ground truth returned by official inference is ignored for HBEC fusion.

Recommended experiment order:

- disabled equivalence check.
- enabled with `evidence_source=late_fusion_reinfer`.
- enabled refine only.
- enabled refine + novel.
- enabled refine + novel + suppress.
