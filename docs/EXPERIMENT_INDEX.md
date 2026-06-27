# Experiment Index

| Version | Method | Status | Main Idea | Target Metric | Baseline Compared | Result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HEAL-XLab-v1 | HBEC | Implemented, not evaluated | object-state / hypothesis-space Bayesian evidence communication | final_infer use_cav2 AP@0.7 | HEAL_m1_based | Not evaluated | Need disabled fallback check, then enabled final_infer test |
| HEAL-XLab-v1.1 | HBEC evidence extraction | Implemented, not evaluated | object-level official re-inference evidence for HBEC refine/novel/suppress | final_infer use_cav2 AP@0.7 | HEAL-XLab-v1 | Not evaluated | Run disabled equivalence, then enabled late_fusion_reinfer experiments |

## Baseline

HEAL_m1_based final_infer:

| Setting | AP@0.3 | AP@0.5 | AP@0.7 |
| --- | ---: | ---: | ---: |
| use_cav1 | 0.7870 | 0.7694 | 0.6726 |
| use_cav2 | 0.8369 | 0.8181 | 0.7248 |
| use_cav3 | 0.8842 | 0.8726 | 0.8044 |
| use_cav4 | 0.8846 | 0.8730 | 0.8045 |
