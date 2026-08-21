# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import os
import sys
from collections import OrderedDict
import glob
import re

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    resolve_v5_quality_safe_config,
    resolve_v6_residual_safe_config,
)


class DualSpaceMergeConfigError(RuntimeError):
    """Raised before merge_final writes a checkpoint with incompatible configs."""


_MERGE_MODALITIES = ('m2', 'm3', 'm4')

def get_model_path_from_dir(model_dir):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise "No checkpoint!"
        
        return os.path.join(save_dir, f'net_epoch{initial_epoch_}.pth')

    file_list = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))

    if len(file_list):
        assert len(file_list) == 1
        model_path = file_list[0]
    else:
        model_path = findLastCheckpoint(model_dir)

    print(f"find {model_path}.")
    
    return model_path


def rename_to_new_version(checkpoint_path):
    # stage1 model to new vesrion
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        # 将 'model.model' 替换为 'channel_align.model'
        new_key = key.replace('model.model', 'channel_align.model')
        new_key = new_key.replace('model.warpnet', 'warpnet')
        new_state_dict[new_key] = old_state_dict[key]


    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def remove_m4_trunk(checkpoint_path):
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        if key.startswith("encoder_m4.camencode.trunk") or \
            key.startswith('encoder_m4.camencode.final_conv') or \
            key.startswith("encoder_m4.camencode.layer3"):
            continue

        new_state_dict[key] = old_state_dict[key]

    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def merge_dict(single_model_dict, stage1_model_dict):
    merged_dict = OrderedDict()
    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict


def apply_dual_space_merge_ownership(merged_dict, ordered_stage_dicts):
    """Overlay feature-aware Dual-Space keys using explicit ownership.

    ``ordered_stage_dicts`` must follow the documented merge_final order:
    Stage2 m2, Stage2 m3, Stage2 m4, then Stage1 m1.  Legacy checkpoints with
    no ``dual_space_`` keys return through unchanged.
    """
    has_dual_space = any(
        key.startswith('dual_space_')
        for state_dict in ordered_stage_dicts
        for key in state_dict
    )
    if not has_dual_space:
        return merged_dict
    if len(ordered_stage_dicts) != 4:
        raise RuntimeError(
            'DS-V1 merge_final requires checkpoints ordered as m2, m3, m4, m1'
        )

    result = OrderedDict(merged_dict)
    stage2_by_modality = {
        'm2': ordered_stage_dicts[0],
        'm3': ordered_stage_dicts[1],
        'm4': ordered_stage_dicts[2],
    }
    stage1_m1 = ordered_stage_dicts[3]
    shared_owners = (
        ('dual_space_shared_object_encoder.', 'shared_object_encoder'),
        ('dual_space_shared_geometry_encoder.', 'shared_geometry_encoder'),
        ('dual_space_shared_object_refiner.', 'shared_object_refiner'),
        ('dual_space_shared_context_encoder.', 'shared_context_encoder'),
        ('dual_space_shared_multiscale_fusion.', 'shared_multiscale_fusion'),
        ('dual_space_shared_scale_gate.', 'shared_scale_gate'),
        ('dual_space_shared_quality_head.', 'shared_quality_head'),
    )
    shared_prefixes = tuple(prefix for prefix, _ in shared_owners)

    print('[DualSpace Merge]')
    for prefix, label in shared_owners:
        if not any(
            key.startswith(prefix)
            for state_dict in ordered_stage_dicts
            for key in state_dict
        ):
            continue
        _replace_owned_prefix(result, stage1_m1, prefix, 'stage1/m1')
        print('%s <- stage1/m1' % label)

    stage1_shared_keys = {
        key for key in stage1_m1 if key.startswith(shared_prefixes)
    }
    for modality, source in stage2_by_modality.items():
        source_shared_keys = {
            key for key in source if key.startswith(shared_prefixes)
        }
        # Real Stage2 checkpoints retain the frozen shared backend.  When it
        # is present, require the exact current-profile key contract.  Tiny
        # adapter-only synthetic dictionaries remain valid merge fixtures.
        if source_shared_keys and source_shared_keys != stage1_shared_keys:
            missing = sorted(stage1_shared_keys - source_shared_keys)
            extra = sorted(source_shared_keys - stage1_shared_keys)
            details = []
            if missing:
                details.append('missing shared keys: %s' % ', '.join(missing))
            if extra:
                details.append('unexpected shared keys: %s' % ', '.join(extra))
            raise RuntimeError(
                '[DualSpace Merge] stage2/%s profile does not match stage1/m1; %s'
                % (modality, '; '.join(details))
            )

    require_detail_adapters = any(
        key.startswith((
            'dual_space_shared_object_encoder.',
            'dual_space_shared_geometry_encoder.',
            'dual_space_shared_object_refiner.',
        ))
        for key in stage1_m1
    )
    require_context_adapters = any(
        key.startswith((
            'dual_space_shared_context_encoder.',
            'dual_space_shared_multiscale_fusion.',
            'dual_space_shared_scale_gate.',
        ))
        for key in stage1_m1
    )
    for modality, source in stage2_by_modality.items():
        for adapter_kind in ('object_adapter', 'context_adapter'):
            prefix = 'dual_space_%s_%s.' % (adapter_kind, modality)
            required = (
                require_detail_adapters
                if adapter_kind == 'object_adapter'
                else require_context_adapters
            )
            present = any(key.startswith(prefix) for key in source)
            if not required and not present:
                continue
            _replace_owned_prefix(result, source, prefix, 'stage2/%s' % modality)
            print('%s_%s <- stage2/%s' % (
                adapter_kind, modality, modality
            ))
    return result


def _replace_owned_prefix(destination, source, prefix, owner):
    owned_keys = [key for key in source if key.startswith(prefix)]
    if not owned_keys:
        raise RuntimeError(
            '[DualSpace Merge] %s has no required keys for %s' % (owner, prefix)
        )
    for key in [key for key in destination if key.startswith(prefix)]:
        del destination[key]
    for key in owned_keys:
        destination[key] = source[key]
    
def merge_and_save(single_model_dir, stage1_model_dir, output_model_dir):
    single_model_path = get_model_path_from_dir(single_model_dir)
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    merged_dict = merge_dict(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)

def merge_and_save_final(aligned_model_dir_list, output_model_dir):
    """
    aligned_model_dir_list:
        e.g. [m2_ALIGNTO_m1_model_dir, m3_ALIGNTO_m1_model_dir, m4_ALIGNTO_m1_model_dir, m1_collaboration_base_dir]

    output_model_dir:
        model_dir.
    """
    validate_dual_space_merge_config_contract(
        aligned_model_dir_list, output_model_dir
    )

    final_dict = OrderedDict()
    ordered_stage_dicts = []
    for aligned_model_dir in aligned_model_dir_list:
        aligned_model_path = get_model_path_from_dir(aligned_model_dir)
        model_dict = torch.load(aligned_model_path, map_location='cpu')
        ordered_stage_dicts.append(model_dict)
        final_dict = merge_dict(final_dict, model_dict)

    final_dict = apply_dual_space_merge_ownership(
        final_dict, ordered_stage_dicts
    )

    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(final_dict, output_model_path)


def validate_dual_space_merge_config_contract(
    aligned_model_dir_list, output_model_dir
):
    """Validate parameter-free extension configs before ``merge_final``.

    The official merge order starts with Stage2 m2, m3, and m4. Legacy runs
    with missing or disabled V5/V6 extensions return without requiring a final
    config. V5 is training-only and is compared across Stage2. V6 changes the
    inference forward and must also match the explicitly supplied final config.
    """
    if len(aligned_model_dir_list) < len(_MERGE_MODALITIES):
        return {'v5_quality_safe': False, 'v6_residual_safe': False}

    stage2_configs = {}
    stage2_paths = {}
    for modality, model_dir in zip(
        _MERGE_MODALITIES, aligned_model_dir_list[:3]
    ):
        path = os.path.join(model_dir, 'config.yaml')
        stage2_paths[modality] = path
        stage2_configs[modality] = _load_optional_merge_config(path, modality)

    v5_configs = {
        modality: _resolve_merge_extension(
            config, resolve_v5_quality_safe_config, modality, 'V5'
        )
        for modality, config in stage2_configs.items()
    }
    v6_configs = {
        modality: _resolve_merge_extension(
            config, resolve_v6_residual_safe_config, modality, 'V6'
        )
        for modality, config in stage2_configs.items()
    }
    v5_enabled = any(config['enabled'] for config in v5_configs.values())
    v6_enabled = any(config['enabled'] for config in v6_configs.values())
    if not v5_enabled and not v6_enabled:
        return {'v5_quality_safe': False, 'v6_residual_safe': False}

    for modality, config in stage2_configs.items():
        if config is None:
            raise DualSpaceMergeConfigError(
                '[DualSpace Merge Config Error]\n'
                'Stage2/%s config.yaml is required because a Dual-Space '
                'merge extension is enabled: %s' % (
                    modality, stage2_paths[modality]
                )
            )

    if v5_enabled:
        _require_stage2_extension_match(v5_configs, 'V5 quality-safe')

    if v6_enabled:
        canonical_v6 = _require_stage2_extension_match(
            v6_configs, 'V6 residual-safe'
        )
        final_path = os.path.join(output_model_dir, 'config.yaml')
        final_config = _load_required_final_config(final_path)
        final_v6 = _resolve_merge_extension(
            final_config,
            resolve_v6_residual_safe_config,
            'final_infer',
            'V6',
        )
        if not final_v6['enabled']:
            raise DualSpaceMergeConfigError(
                '[DualSpace Merge Config Error]\n'
                'V6 residual-safe is enabled in Stage2 but disabled in final '
                'inference config.\nUse the matching merged_infer.yaml before '
                'running merge_final.'
            )
        if final_v6 != canonical_v6:
            raise DualSpaceMergeConfigError(
                '[DualSpace Merge Config Error]\n'
                'V6 configuration mismatch between stage2/m2 and final_infer.'
            )
        print(
            '[DualSpace Merge Config]\n'
            'V6 residual-safe contract validated for stage2 -> merged inference'
        )

    return {
        'v5_quality_safe': v5_enabled,
        'v6_residual_safe': v6_enabled,
    }


def _load_optional_merge_config(path, label):
    if not os.path.isfile(path):
        return None
    try:
        return load_yaml(path, None)
    except Exception as error:
        raise DualSpaceMergeConfigError(
            '[DualSpace Merge Config Error]\n'
            'Could not read Stage2/%s config: %s' % (label, error)
        ) from error


def _load_required_final_config(path):
    if not os.path.isfile(path):
        raise DualSpaceMergeConfigError(
            '[DualSpace Merge Config Error]\n'
            'V6 residual-safe is enabled in Stage2 but final inference '
            'config.yaml is missing: %s\nUse the matching merged_infer.yaml '
            'before running merge_final.' % path
        )
    try:
        return load_yaml(path, None)
    except Exception as error:
        raise DualSpaceMergeConfigError(
            '[DualSpace Merge Config Error]\n'
            'Could not read final inference config: %s' % error
        ) from error


def _resolve_merge_extension(config, resolver, label, extension):
    dual_space = None
    if isinstance(config, dict):
        model = config.get('model')
        args = model.get('args') if isinstance(model, dict) else None
        if isinstance(args, dict):
            dual_space = args.get('dual_space')
    if dual_space is not None and not isinstance(dual_space, dict):
        raise DualSpaceMergeConfigError(
            '[DualSpace Merge Config Error]\n'
            '%s model.args.dual_space must be a mapping' % label
        )
    try:
        return resolver(dual_space)
    except Exception as error:
        raise DualSpaceMergeConfigError(
            '[DualSpace Merge Config Error]\n'
            '%s %s configuration is invalid: %s'
            % (label, extension, error)
        ) from error


def _require_stage2_extension_match(configs, label):
    canonical = configs['m2']
    for modality in _MERGE_MODALITIES[1:]:
        if configs[modality] != canonical:
            raise DualSpaceMergeConfigError(
                '[DualSpace Merge Config Error]\n'
                '%s configuration mismatch between stage2/m2 and stage2/%s.'
                % (label, modality)
            )
    return canonical


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'rename_to_new_version':
        checkpoint_path = sys.argv[2]
        rename_to_new_version(checkpoint_path)
    elif func == 'remove_m4_trunk':
        checkpoint_path = sys.argv[2]
        remove_m4_trunk(checkpoint_path)
    elif func == 'merge':
        single_model_dir = sys.argv[2]
        stage1_model_dir = sys.argv[3]
        output_model_dir = sys.argv[4]
        merge_and_save(single_model_dir, stage1_model_dir, output_model_dir)
    elif func == 'merge_final': 
        merge_and_save_final(sys.argv[2:-1], sys.argv[-1])
    else:
        raise "This function not implemented"
