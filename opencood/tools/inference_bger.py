# -*- coding: utf-8 -*-
"""BGER inference entry.

A standalone variant of opencood/tools/inference.py for the BGER line. It
adds, on top of the standard intermediate-style flow:

- ``--box_merge``: merge the collaborator box messages exposed in
  ``output_dict['bger']`` with the ego detections at the box level
  (rotated NMS). Combined with ``bger.mode: box_merge_only`` in the yaml,
  this evaluates the classic late-fusion control group under the exact
  same communication content as BGER.
- Communication volume accounting averaged over the test set
  (``comm_bytes_boxes`` vs ``comm_bytes_feature_equiv``), for the
  accuracy-bandwidth pareto analysis.

The official inference.py is untouched.
"""

import argparse
import importlib
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils, eval_utils
from opencood.utils.common_utils import update_dict

torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="BGER inference")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='training log path containing config.yaml and checkpoints')
    parser.add_argument('--box_merge', action='store_true',
                        help='merge collaborator box messages with ego detections '
                             'at the box level (late-fusion control group)')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization; <=0 disables')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range [-x, +x, -y, +y]")
    parser.add_argument('--note', default="", type=str)
    return parser.parse_args()


def inference_bger_fusion(batch_data, model, dataset, box_merge=False):
    """Intermediate-style inference that understands BGER outputs."""
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data, output_dict)

    bger_info = output_dict['ego'].get('bger', {})

    if box_merge and bger_info:
        collab_boxes = bger_info["collab_boxes_ego_frame"][0]
        collab_scores = bger_info["collab_box_scores"][0]
        if collab_boxes is not None and collab_boxes.shape[0] > 0:
            order = dataset.post_processor.params['order']
            nms_thresh = dataset.post_processor.params['nms_thresh']
            collab_corners = box_utils.boxes_to_corners_3d(
                collab_boxes.float(), order=order
            )
            collab_scores = collab_scores.float()
            if pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
                all_corners = collab_corners
                all_scores = collab_scores
            else:
                all_corners = torch.cat([pred_box_tensor, collab_corners], dim=0)
                all_scores = torch.cat([pred_score, collab_scores], dim=0)
            keep_index = box_utils.nms_rotated(all_corners, all_scores, nms_thresh)
            pred_box_tensor = all_corners[keep_index]
            pred_score = all_scores[keep_index]

    return {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "bger_info": bger_info,
    }


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [
            x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        })
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    if opt.box_merge:
        opt.note += "_boxmerge"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(303)
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    comm_bytes_boxes_sum = 0.0
    comm_bytes_feature_sum = 0.0
    comm_frames = 0

    infer_info = "bger" + opt.note

    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            infer_result = inference_bger_fusion(
                batch_data, model, opencood_dataset, box_merge=opt.box_merge
            )

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            for iou_thresh in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                           gt_box_tensor, result_stat, iou_thresh)

            bger_info = infer_result.get('bger_info') or {}
            if "comm_bytes_boxes" in bger_info:
                comm_bytes_boxes_sum += float(bger_info["comm_bytes_boxes"])
                comm_bytes_feature_sum += float(bger_info["comm_bytes_feature_equiv"])
                comm_frames += 1

            if opt.save_vis_interval > 0 and (i % opt.save_vis_interval == 0) \
                    and (pred_box_tensor is not None or gt_box_tensor is not None):
                from opencood.visualization import simple_vis
                infer_result.update({'score_tensor': pred_score})
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                     batch_data['ego']['origin_lidar'][0],
                                     hypes['postprocess']['gt_range'],
                                     vis_save_path,
                                     method='bev',
                                     left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)

    if comm_frames > 0:
        avg_boxes = comm_bytes_boxes_sum / comm_frames
        avg_feature = comm_bytes_feature_sum / comm_frames
        ratio = avg_boxes / avg_feature if avg_feature > 0 else 0.0
        print("BGER communication accounting over %d frames:" % comm_frames)
        print(" - avg box-message bytes / frame:        %.1f" % avg_boxes)
        print(" - avg feature-equivalent bytes / frame: %.1f" % avg_feature)
        print(" - bandwidth ratio (boxes / features):   %.6f" % ratio)


if __name__ == '__main__':
    main()
