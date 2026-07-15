"""CPU smoke coverage for the read-only PACT-CBEA real geometry auditor."""
from __future__ import print_function

import ast
import csv
import json
import os
import subprocess
import sys
import tempfile
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The repository utility imports icecream even though this CPU geometry smoke
# only uses normalize_pairwise_tfm.  Keep the compatibility shim process-local.
_installed_smoke_stubs = []
if "icecream" not in sys.modules:
    icecream_stub = types.ModuleType("icecream")
    icecream_stub.ic = lambda *args, **kwargs: args[0] if args else None
    sys.modules["icecream"] = icecream_stub
    _installed_smoke_stubs.append("icecream")
if "pyquaternion" not in sys.modules:
    quaternion_stub = types.ModuleType("pyquaternion")
    quaternion_stub.Quaternion = object
    sys.modules["pyquaternion"] = quaternion_stub
    _installed_smoke_stubs.append("pyquaternion")
if "shapely" not in sys.modules:
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry_stub = types.ModuleType("shapely.geometry")
    shapely_geometry_stub.Polygon = object
    shapely_stub.geometry = shapely_geometry_stub
    sys.modules["shapely"] = shapely_stub
    sys.modules["shapely.geometry"] = shapely_geometry_stub
    _installed_smoke_stubs.extend(["shapely.geometry", "shapely"])

from opencood.tools.audit_pact_cbea_real_geometry import (
    CSV_FIELDS,
    _inverse_affine,
    _capture_model_geometry,
    _write_outputs,
    build_candidate_affine,
    extract_aligned_geometry_fields,
    resolve_checkpoint_path,
    synthetic_agent_order_check,
    warp_feature_and_validity,
)
from opencood.models.sub_modules.pact_cbea_evidence_routed import (
    PACTCBEAEvidenceGeometryAligner,
)

for _stub_name in _installed_smoke_stubs:
    sys.modules.pop(_stub_name, None)


AUDIT_PATH = os.path.join(ROOT, "opencood", "tools", "audit_pact_cbea_real_geometry.py")
PROTECTED_MARKERS = [
    "heter_pyramid_collab_pact_cbea.py",
    "heter_pyramid_collab_pact_cbea_evidence_routed.py",
    "heter_pyramid_collab_pact_cbea_aligned_uniform.py",
    "heter_pyramid_collab_pact_cbea_packet_nojoint.py",
    "heter_pyramid_collab_pact_cbea_box_packet_nojoint.py",
    "inference_utils.py",
]


def _assert_close(left, right, message, atol=1e-5):
    if not torch.allclose(left, right, atol=atol, rtol=0.0):
        raise AssertionError(message)


def _pairwise(count=2):
    matrix = torch.eye(4).view(1, 1, 4, 4).repeat(count, count, 1, 1)
    # source 1 is +2 m in ego coordinates; the reverse is -2 m.
    matrix[1, 0, 0, 3] = 2.0
    matrix[0, 1, 0, 3] = -2.0
    return matrix


def _translation_affine(tx, ty, h=8, w=8):
    affine = torch.eye(4).view(1, 1, 1, 4, 4)
    affine[0, 0, 0, 0, 3] = tx
    affine[0, 0, 0, 1, 3] = ty
    return build_candidate_affine(affine[0], "A_ego_source", h, w, 1.0)


def _assert_python38(path):
    source = open(path, "r", encoding="utf-8").read()
    try:
        tree = ast.parse(source, filename=path, feature_version=8)
    except TypeError:
        # Python 3.8 itself already parses with its native grammar.
        tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            raise AssertionError("walrus syntax is not allowed for Python 3.8 compatibility policy")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            raise AssertionError("dict union/operator | is not allowed")


def _assert_only_new_files_changed():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, universal_newlines=True).splitlines()
    for path in changed:
        if any(path.endswith(marker) for marker in PROTECTED_MARKERS):
            raise AssertionError("protected production file changed: {}".format(path))


def _identity_affine(batch, agent_count):
    return torch.eye(2, 3).view(1, 1, 1, 2, 3).repeat(batch, agent_count, agent_count, 1, 1)


def _assert_aligner_wrapper_contract():
    feature = torch.arange(32, dtype=torch.float32).view(2, 2, 4, 2)
    logits = torch.zeros((2, 1, 4, 2))
    uncertainty = torch.ones((2, 1, 4, 2))
    descriptor = torch.zeros((2, 1, 4, 2))
    record_len = torch.tensor([2])
    affine = _identity_affine(1, 2)
    aligner = PACTCBEAEvidenceGeometryAligner(align_corners=False)
    original_result = aligner(feature, logits, uncertainty, descriptor, record_len, affine)
    expected_keys = {"feature", "heatmap_logits", "uncertainty", "descriptor", "validity", "debug"}
    if not isinstance(original_result, dict) or set(original_result.keys()) != expected_keys:
        raise AssertionError("real PACT geometry aligner contract changed")
    extracted = extract_aligned_geometry_fields(original_result)
    if set(extracted.keys()) != {"feature", "validity"}:
        raise AssertionError("audit extraction did not select the documented fields")

    class Pyramid(object):
        def forward_single(self, tensor):
            return tensor, []

    class RecordingAligner(object):
        def __init__(self):
            self.last_result = None

        def forward(self, *unused_args):
            self.last_result = {
                "feature": original_result["feature"],
                "heatmap_logits": original_result["heatmap_logits"],
                "uncertainty": original_result["uncertainty"],
                "descriptor": original_result["descriptor"],
                "validity": original_result["validity"],
                "debug": original_result["debug"],
            }
            return self.last_result

    class Model(object):
        def __init__(self):
            self.pyramid_backbone = Pyramid()
            self.pact_geometry_aligner = RecordingAligner()

    model = Model()
    capture, original_single, original_geometry = _capture_model_geometry(model)
    try:
        wrapped_result = model.pact_geometry_aligner.forward(
            feature, logits, uncertainty, descriptor, record_len, affine
        )
        if wrapped_result is not model.pact_geometry_aligner.last_result:
            raise AssertionError("geometry wrapper replaced the original result object")
        if type(wrapped_result) is not dict or set(wrapped_result.keys()) != expected_keys:
            raise AssertionError("geometry wrapper changed return type or keys")
        _assert_close(capture["aligned_feature"], original_result["feature"],
                      "wrapper captured incorrect aligned feature")
        _assert_close(capture["aligned_validity"], original_result["validity"],
                      "wrapper captured incorrect validity")
    finally:
        model.pyramid_backbone.forward_single = original_single
        model.pact_geometry_aligner.forward = original_geometry

    for invalid_result in ((), [], torch.zeros(1), {"feature": feature}):
        try:
            extract_aligned_geometry_fields(invalid_result)
        except RuntimeError as error:
            if not str(error):
                raise AssertionError("invalid geometry result error was not descriptive")
        else:
            raise AssertionError("invalid geometry result was silently accepted")


def _assert_checkpoint_resolution():
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        model_dir = os.path.join(directory, "model")
        os.makedirs(model_dir)
        default_path = resolve_checkpoint_path(model_dir)
        relative_path = resolve_checkpoint_path(model_dir, "net_epoch9.pth")
        already_prefixed = os.path.relpath(
            os.path.join(model_dir, "net_epoch8.pth"), ROOT
        )
        already_prefixed_path = resolve_checkpoint_path(model_dir, already_prefixed)
        absolute_path = resolve_checkpoint_path(model_dir, os.path.join(model_dir, "net_epoch7.pth"))
        if default_path != os.path.join(model_dir, "net_epoch1.pth"):
            raise AssertionError("default checkpoint path was not model_dir/net_epoch1.pth")
        if relative_path != os.path.join(model_dir, "net_epoch9.pth"):
            raise AssertionError("relative checkpoint path was not resolved once")
        if already_prefixed_path != os.path.join(model_dir, "net_epoch8.pth"):
            raise AssertionError("already-prefixed checkpoint path was duplicated")
        if absolute_path != os.path.join(model_dir, "net_epoch7.pth"):
            raise AssertionError("absolute checkpoint path changed")


def main():
    torch.manual_seed(7)
    feature = torch.zeros((1, 1, 8, 8))
    feature[0, 0, 2, 3] = 1.0
    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    identity_feature, identity_validity, identity_grid = warp_feature_and_validity(feature, identity)
    _assert_close(identity_feature, feature, "identity feature changed")
    _assert_close(identity_validity, torch.ones_like(identity_validity), "identity validity changed")

    positive = _translation_affine(1.0, 0.0)
    negative = _translation_affine(-1.0, 0.0)
    positive_feature, positive_validity, positive_grid = warp_feature_and_validity(feature, positive)
    negative_feature, negative_validity, negative_grid = warp_feature_and_validity(feature, negative)
    if torch.allclose(positive_feature, negative_feature):
        raise AssertionError("positive/negative x translations are indistinguishable")
    y_positive = _translation_affine(0.0, 1.0)
    _, _, y_grid = warp_feature_and_validity(feature, y_positive)
    if torch.allclose(positive_grid, y_grid):
        raise AssertionError("x/y translations are indistinguishable")

    rotation = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]])
    rotated, rotated_validity, _ = warp_feature_and_validity(feature, rotation)
    if torch.allclose(rotated, feature):
        raise AssertionError("90 degree rotation did not change asymmetric feature")
    if not bool((rotated_validity >= 0).all() and (rotated_validity <= 1).all()):
        raise AssertionError("rotation validity left [0, 1]")

    pairwise = _pairwise()
    affine_a = build_candidate_affine(pairwise, "A_ego_source", 8, 8, 1.0)
    affine_b = build_candidate_affine(pairwise, "B_source_ego", 8, 8, 1.0)
    if torch.allclose(affine_a, affine_b):
        raise AssertionError("[ego, source] and [source, ego] must differ for non-identity transform")
    round_trip, _, _ = warp_feature_and_validity(
        warp_feature_and_validity(feature.repeat(2, 1, 1, 1), affine_a)[0], _inverse_affine(affine_a))
    if not torch.isfinite(round_trip).all():
        raise AssertionError("inverse round trip produced non-finite values")
    expected_validity = torch.nn.functional.grid_sample(
        torch.ones_like(feature), positive_grid, mode="bilinear", padding_mode="zeros",
        align_corners=False).clamp(0.0, 1.0)
    _assert_close(positive_validity, expected_validity,
                  "feature and validity do not use the same sampling grid")
    if not bool((positive_validity > 0).any() and (positive_validity < 1).any()):
        raise AssertionError("translation did not produce continuous validity boundary")

    # record_len=[2,3] isolation uses separate pairwise blocks and no cross-scene tensor.
    first = torch.full((2, 1, 4, 4), 2.0)
    second = torch.full((3, 1, 4, 4), 7.0)
    first_out, _, _ = warp_feature_and_validity(first, torch.eye(2, 3).view(1, 2, 3).repeat(2, 1, 1))
    second_out, _, _ = warp_feature_and_validity(second, torch.eye(2, 3).view(1, 2, 3).repeat(3, 1, 1))
    _assert_close(first_out, first, "first record_len scene changed")
    _assert_close(second_out, second, "second record_len scene changed")
    if float(first_out.mean()) == float(second_out.mean()):
        raise AssertionError("record_len scene isolation failed")

    order = synthetic_agent_order_check()
    if not order["restored_matches_input"]:
        raise AssertionError("modal grouping/restoration order changed")

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        report = {"scenes": [], "synthetic_agent_order": order}
        rows = [dict((field, "x") for field in CSV_FIELDS)]
        json_path, csv_path = _write_outputs(directory, report, rows)
        loaded = json.load(open(json_path, "r", encoding="utf-8"))
        with open(csv_path, "r", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if "scenes" not in loaded or header != CSV_FIELDS:
            raise AssertionError("JSON/CSV output fields are incomplete")

    _assert_aligner_wrapper_contract()
    _assert_checkpoint_resolution()
    _assert_python38(AUDIT_PATH)
    _assert_python38(os.path.abspath(__file__))
    _assert_only_new_files_changed()
    print("identity/translation/rotation: OK")
    print("candidate directions and inverse round-trip: OK")
    print("validity grid, record_len isolation, and agent order: OK")
    print("JSON/CSV fields and Python 3.8 AST: OK")
    print("real aligner wrapper contract and checkpoint resolution: OK")
    print("PACT_CBEA_REAL_GEOMETRY_AUDIT_SMOKE_PASS")


if __name__ == "__main__":
    main()
