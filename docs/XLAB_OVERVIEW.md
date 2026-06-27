# HEAL-XLab Overview

HEAL-XLab is an experimental plugin layer for HEAL. It is designed for bold post-training ideas while preserving the official HEAL path by default.

Core rules:

- XLab is disabled unless yaml explicitly sets `xlab.enabled: true`.
- Each method also has its own method-level switch, currently `xlab.hbec.enabled`.
- When disabled, the hook returns official `pred_box_tensor`, `pred_score`, and `gt_box_tensor` unchanged.
- XLab does not bypass raw camera features, modify training, or use ground truth for fusion.
- Experimental switches live under the single `xlab` yaml subtree.

Current method:

- Version: `HEAL-XLab-v1`
- Method: `HBEC`
- Full name: Hypothesis-guided Bayesian Evidence Communication
- Stage: final-infer post-process hook
- Status: implemented, not evaluated

