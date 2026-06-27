"""Debug metric recording for XLab."""

import json
import os
import time


class XLabMetricsRecorder:
    """Append per-frame HBEC metrics as jsonl records."""

    def __init__(self, xlab_cfg, infer_context=None):
        self.cfg = xlab_cfg or {}
        self.infer_context = infer_context or {}

    def default_record(self):
        return {
            "timestamp": time.time(),
            "hbec_enabled": False,
            "matched_count": 0,
            "refined_count": 0,
            "novel_count": 0,
            "suppressed_count": 0,
            "payload_bytes_est": 0,
            "fallback_reason": "",
            "nms_status": "not_applied",
        }

    def write(self, record):
        if not self.cfg.get("debug", True):
            return
        merged = self.default_record()
        merged.update(record or {})
        merged.update({
            "frame_id": self.infer_context.get("frame_id"),
            "infer_info": self.infer_context.get("infer_info"),
            "fusion_method": self.infer_context.get("fusion_method"),
            "use_cav": self.infer_context.get("use_cav"),
        })
        root = self.infer_context.get("model_dir") or "."
        debug_dir = self.cfg.get("debug_dir", "xlab_debug")
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(root, debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "hbec_debug.jsonl")
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(merged, sort_keys=True) + "\n")

