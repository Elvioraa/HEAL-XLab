"""Post-process hook entry points for XLab."""

import warnings

from opencood.xlab.config import hbec_is_enabled, safe_get_xlab_cfg
from opencood.xlab.metrics import XLabMetricsRecorder


def apply_xlab_postprocess_hook(
    pred_box_tensor,
    pred_score,
    gt_box_tensor,
    batch_data=None,
    model=None,
    hypes=None,
    infer_context=None,
):
    """Optionally apply XLab post-processing, otherwise return official outputs."""
    xlab_cfg = safe_get_xlab_cfg(hypes)
    if not xlab_cfg.get("enabled", False):
        return pred_box_tensor, pred_score, gt_box_tensor

    recorder = XLabMetricsRecorder(xlab_cfg, infer_context)
    if not hbec_is_enabled(xlab_cfg):
        recorder.write({"fallback_reason": "hbec_disabled"})
        return pred_box_tensor, pred_score, gt_box_tensor

    if pred_box_tensor is None or pred_score is None:
        warnings.warn("XLab HBEC fallback: missing official predictions.")
        recorder.write({"hbec_enabled": True, "fallback_reason": "missing_official_prediction"})
        return pred_box_tensor, pred_score, gt_box_tensor

    try:
        from opencood.xlab.hbec.engine import HBECPostProcessor

        engine = HBECPostProcessor(xlab_cfg, recorder=recorder)
        return engine(
            pred_box_tensor=pred_box_tensor,
            pred_score=pred_score,
            gt_box_tensor=gt_box_tensor,
            batch_data=batch_data,
            model=model,
            hypes=hypes,
            infer_context=infer_context,
        )
    except Exception as exc:
        if xlab_cfg.get("hbec", {}).get("safety", {}).get("fallback_on_error", True):
            warnings.warn("XLab HBEC fallback after error: %s" % exc)
            recorder.write({"hbec_enabled": True, "fallback_reason": "exception:%s" % type(exc).__name__})
            return pred_box_tensor, pred_score, gt_box_tensor
        raise

