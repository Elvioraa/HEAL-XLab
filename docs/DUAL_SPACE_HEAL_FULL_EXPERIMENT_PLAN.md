# HEAL Dual-Space 完整实验方案

## 1. 文档定位

本文档记录 HEAL-XLab 中 **Open Heterogeneous Dual-Space Backward Adaptation** 的研究动机、代码架构、真实 tensor contract、训练与 merge 规则、正式实验配置、消融路线、评价指标和风险诊断。后续会话应以本文档和同一源码 commit 为交接依据，而不是重新推测实现意图。

核心开发原则是：

- 代码一次实现 Full Framework。
- 科学实验通过 YAML 按 HEAL → DS-V1 → DS-V2 → DS-V3 → DS-V4 逐层启用。
- profile/version 只是实验标签，不驱动代码分支。
- 所有行为只由 `model.args.dual_space` 下的显式开关和 mode 决定。
- 新功能关闭时不构造其参数，不产生其 state key，也不执行其计算。

当前主方法到 `Stage1 → Stage2 → merge_final → inference` 为止，没有额外 20 epoch heterogeneous Stage3。

## 2. 研究背景

HEAL 原方法通过 modality-specific Encoder、Backbone 和 Aligner，把异构传感器映射到一个 **Common BEV Space**，再由共享 PyramidFusion 完成 scene-level cooperative discovery。这解决了开放异构系统中的场景级兼容问题，但没有显式保证：同一个目标在不同 modality/CAV 上具有可比较、可共享的 object-level geometric representation。

因此，本项目引入第二公共空间：

- **Space-I: Common BEV Space**，回答 Where / What，负责场景级协作发现。
- **Space-II: Common Object Space**，回答 Exactly where / how，负责目标级几何精修与跨 agent 共识。

两个空间不是平行孤立的分支。Object loss 通过可微 ROI sampling 反向传播到 Common BEV feature，再进入当前可训练模态的 Aligner、Backbone 和 Encoder，形成：

```text
scene representation → object representation
object supervision   → BEV representation
```

的闭环。

## 3. 历史观测与动机

当前 merged Stage2 历史观测如下。它们用于提出问题，不代表因果结论：

| CAV | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| CAV1 | 0.7878 | 0.7700 | 0.6641 |
| CAV1-2 | 0.8342 | 0.8180 | 0.7149 |
| CAV1-3 | 0.8801 | 0.8702 | 0.7955 |
| CAV1-4 | 0.8793 | 0.8695 | 0.7950 |

第 4 辆 CAV 加入后，AP@0.7 的边际变化为 `-0.0005`。该差异很小，不能单独证明 m4 有问题，但提示“更多 agent 的信息不一定天然都是正贡献”。这构成了 object-specific、agent-specific Quality-aware Geometry Consensus 的合理动机。

项目研究目标是让 CAV1-4 的 AP@0.7 达到约 0.83 或更高，同时 AP@0.3/AP@0.5 不出现不可接受的退化。该目标是实验方向，不是代码验收条件。

## 4. 旧 PACT/Object Stage3 的教训

历史 PACT/Object Stage3 使用 proposal-conditioned per-agent ROI、box residual、log variance 和 per-dimension precision fusion。旧 ROI 路径曾出现 `proposal × agent × full BEV` 展开，validation 尝试过一次约 21.25 GiB 的临时分配，并伴随复杂 variance loss、训练模式和验证显存问题。

新 Dual-Space 主线明确避免：

- 不建立 feature cache。
- 不做 proposal-by-full-map expansion。
- 不预测 7D variance 或 log variance。
- 不做 per-dimension precision fusion。
- 不引入大型 Transformer/Cross-Attention。
- 不进行 m2/m3/m4 联合训练。
- 不新增重型最终 Stage3。

旧 PACT/Object Stage3、ROI sampler、variance/precision fusion 和 trainer 保持独立历史 namespace，不作为新方法依赖。

## 5. 总体架构

```text
heterogeneous sensors
        |
        v
Modality Encoder / Backbone / Aligner
        |
        v
========================================================
Space-I: Common BEV Space
per-agent common feature, identity retained
========================================================
        |                              |
        |                              +--> pre-fusion Detail/Context evidence
        v
shared HEAL PyramidFusion
        |
        v
fused detector proposals
        |
        +--> optional Remote Proposal Rescue (inference only)
        |
        v
candidate pool
        |
        v
chunked rotated ROI sampling (Detail [+ Context])
        |
        v
modality-specific object adapters
        |
        v
shared object encoder + geometry encoder
        |
        v
========================================================
Space-II: Common Object Space, fixed 128D interface
========================================================
        |
        v
shared geometry refiner: 8D periodic residual
        |
        +--> optional scalar quality q(agent, object)
        |
        v
uniform / quality-weighted geometry consensus
        |
        v
refined boxes
```

## 6. 真实 Common BEV 与多尺度 Tensor Contract

### 6.1 Detail

捕获位置是 `heter_pyramid_collab.py` 中各 modality 已完成 Encoder、modality Backbone、Aligner、camera padding，并在可选 compressor 之后、进入 `PyramidFusion.forward_collab()` 之前的 `heter_feature_2d`。

- Shape: `[sum(A), 64, H, W]`。
- Agent identity: 保留。
- Semantic contract: 所有模态已进入 64-channel Common BEV interface。
- Spatial contract: 尚未跨 agent fusion；object context 构建时按 `[B, ego=0, agent]` affine warp 到 ego。
- 官方 stage1 m1: `H=128, W=256`，物理 stride 为 0.8 m。
- merged 方形范围配置通常为 `H=256, W=256`，物理 stride 仍为 0.8 m。
- DS-V1 Detail 分支及其 state keys 不因 Full Framework 改写。

### 6.2 Context

`PyramidFusion.get_multiscale_feature()` 在任何 cross-agent `weighted_fuse()` 之前已经产生 per-agent shared pyramid features。DS-V2 直接捕获 index 1：

- Shape: `[sum(A), 128, H/2, W/2]`。
- 来源: shared `PyramidFusion.resnet` level 1。
- 物理 stride: 1.6 m。
- Capture: shared pyramid backbone 之后、跨 agent weighted fusion 之前。
- 不重新运行 modality Encoder、sensor backbone 或 Aligner。
- 仅 `multi_scale.enabled=true` 时对该层做 ego warp、Context ROI 和 Context Encoder。

RPR 打开时会临时捕获全部 shared pre-fusion pyramid levels `[64,128,256]`，将每个 agent 的各层 warp 到 ego，再只重跑 shared deblocks/shrink/head。解码完成后不保留这些完整 pyramid tensor。

### 6.3 Transform 方向

`pairwise_t_matrix[b, 0, j]` 与 normalized affine `[b,0,j]` 表示 ego output coordinate 到 agent-j input coordinate 的采样方向。`affine_grid` 使用该方向把 agent feature warp 到 ego。agent-j 原点在 ego 中的位置由该刚体变换的逆矩阵得到，用于 quality distance，不新增通信量。

## 7. Box Convention

Dual-Space 对外统一使用仓库约定：

```text
[x, y, z, h, w, l, yaw]
```

- x 对应 feature W/column，y 对应 H/row。
- yaw 以弧度表示，正 yaw 将局部 +x/length 轴旋向 +y。
- residual 为 `[dx,dy,dz,dlog_l,dlog_w,dlog_h,sin(dyaw),cos(dyaw)]`。
- 平移分别按 proposal 的 l、w、h 归一化。
- yaw decode 使用 `atan2(sin, cos)` 并 wrap 到 `[-pi, pi)`。
- 仅在调用仓库 Stage3 decoder/rotated IoU 时显式转换为 `xyzlwhr`，返回后立即恢复 hwl。

## 8. M1 Efficient Rotated ROI Extractor

- 文件: `opencood/models/sub_modules/dual_space_object_roi.py`。
- 输入: ego-aligned `[A,C,H,W]`、proposal `[N,7]`、可选 support `[A,1,H,W]`。
- 输出: ROI `[N,A,C,Rh,Rw]`、valid `[N,A]`、coverage `[N,A]`。
- 实现: pure PyTorch `grid_sample`, `align_corners=False`, zero padding。
- coverage: ROI bin center 同时位于 normalized `[-1,1]` 且 support 有效的比例。
- 内存: 每次只处理一个 agent 和最多 `chunk_size` proposals；不存在 `[N,A,C,H,W]` full-map tensor。
- Stage1/Stage2: 可微到输入 feature，无可学习参数。
- Inference: ego 无效不会自动丢 proposal；任一 remote valid 即可参与。

## 9. M2 Modality-Specific Object Adapter

- Detail m1: `Identity`，无参数。
- Detail m2/m3/m4: `dual_space_object_adapter_mX`，Residual 1x1 + GroupNorm + GELU，末层 zero-init，初始化严格 identity。
- Context m1: `Identity`。
- Context m2/m3/m4: `dual_space_context_adapter_mX`，相同 residual-safe 结构，输入 channel 为 128。
- Stage1: m1 identity；不存在非必要新模态 adapter 训练。
- Stage2: 只训练当前 active modality 的 Detail/Context adapter。
- Shared backend 冻结，但梯度可穿过 frozen shared module 回到 active adapter 与 modality branch。

## 10. M3 Shared Object Encoder

- Detail 输入: `[M,64,5,5]`，其中 M 为 valid agent-object pair 数。
- 结构: Conv/GN/GELU ×2 → AdaptiveAvgPool → Linear → LayerNorm。
- 输出: `z_detail [M,128]`。
- 所有模态共享；Stage1 建立，Stage2 冻结。
- DS-V1 路径保持原实现，不因 Context 分支而改写。

## 11. M4 Proposal Geometry Encoder

- 输入: normalized `[x,y,z,log(l),log(w),log(h),sin(yaw),cos(yaw)] [M,8]`。
- 归一化范围来自实际 `lidar_range`，不是硬编码。
- 输出: geometry embedding `[M,32]`。
- 所有模态共享；Stage1 训练，Stage2 冻结。

## 12. M5 Shared Geometry Refiner

- 输入: object embedding 128D + geometry embedding 32D。
- 输出: periodic 8D residual。
- 末层 zero-init，因此未经训练时 box residual 为零，decode 等于原 proposal。
- Stage1 训练，Stage2 冻结。
- 不预测 uncertainty/variance，不做 per-dimension weighting。

## 13. M6 Multi-Scale Object Representation

当 `multi_scale.enabled=true`：

```text
Detail ROI 5x5   → Detail Adapter  → Shared Object Encoder  → z_detail  [128]
Context ROI 3x3  → Context Adapter → Shared Context Encoder → z_context [128]
```

正式 DS-V2 使用 `fusion: concat_projection`：

```text
concat(z_detail,z_context) [256]
→ Linear 256→128 → LayerNorm → GELU
→ residual-safe scale
→ z_multi [128]
```

fusion module 的 residual scale 从 0 开始，因此 V1→V2 warm start 时初始输出严格等于 Detail。最终 Common Object Space 始终为 128D，而不是扩展到 256/512D。

可选消融 `fusion: adaptive_gate` 使用 shared tiny scalar gate：

```text
g = sigmoid(MLP([z_detail,z_context]))
z = g*z_detail + (1-g)*z_context
```

gate 初始 bias 使其接近 Detail-only。只保留这一套配置 API，没有重复的 `adaptive_scale_gate.enabled`。

未选择四个 pyramid level 全量 concat，原因是参数、显存、延迟和因果解释成本都会增加；Detail + Context 已覆盖高分辨率边界与大感受野遮挡语义。

## 14. M7 Quality Head

Quality 是每个 agent 对某个 proposal 的几何修正质量，不是 agent global reliability、modality global reliability 或 variance。

输入为：

```text
object embedding        128
geometry embedding       32
ROI coverage              1  (optional explicit switch)
normalized agent distance 1  (optional explicit switch)
total                   162
```

结构为 `Linear → GELU → Linear → sigmoid`，正式 hidden dim 为 64，输出 `q_i ∈ [0,1]`。所有模态共享一个 Quality Head。

target 定义为：

```text
q_target_i = rotated_BEV_IoU(individual_refined_box_i, matched_GT)
```

target 始终 detached。CUDA 正式路径必须调用仓库 `boxes_iou_bev`；若 compiled CUDA extension 不可用则直接报错，不允许 silent approximation。CPU smoke 使用仓库已有 mathematically equivalent pure-PyTorch polygon fallback。

Quality loss 为 valid agent-object pair 上的 `SmoothL1(q_i,q_target_i)`。Stage1 更新 Quality Head；Stage2 Quality Head 参数冻结，但输入梯度继续回到 active modality adapter/BEV branch。

正式 V3 YAML 中 `quality_loss_weight=0.05` 是 **initial hyperparameter, not final tuned value**。

## 15. M8 Geometry Consensus

### Uniform

DS-V1/V2 对 valid residual 的 6 个连续量和 sin/cos yaw 分量做 uniform mean。没有 valid agent 时，box 严格 fallback 到原 proposal。

### Quality-weighted

DS-V3 使用：

```text
w_i = valid_i * q_i
delta = sum(w_i * delta_i) / sum(w_i)
```

默认 `detach_weight_for_consensus=true`，因此 consensus loss 不通过权重捷径直接更新 Quality Head；Quality Head 由独立 target loss 监督。当 quality sum 小于 `min_quality_sum` 时，逐值回退到 uniform consensus。ego q 低而 remote q 高时，remote residual 会自然主导，而不是要求 ego 必须 valid。

## 16. M9 Remote Proposal Rescue

Object Space 本身不负责从无到有发现目标：

- ego miss、remote see、HEAL fused detector 有 proposal：DS-V1/V2/V3 已能处理，不需要 RPR。
- ego miss、remote see、HEAL fused detector 也 miss：没有 candidate 时 Object Space 无法工作，RPR 才补偿 discovery failure。

RPR 为 inference-only、parameter-free policy：

1. 在 shared pre-fusion pyramid 上获得每个 agent 的 ego-aligned dense detector output。
2. 使用仓库 detector decoder、direction correction、range filter 和 rotated NMS。
3. 默认排除 ego local proposal。
4. 过滤 `min_score`，限制 `max_per_agent`。
5. 删除与 fused proposal IoU 超阈值的 remote candidate。
6. 按 score、agent index、原 proposal index 做 deterministic greedy dedup。
7. 最多追加 `max_total_added` 个 candidate。
8. fused score 原值、原顺序保持；rescued candidate 使用最高 remote score。
9. 合并后只对 top-K 做 Object refinement，未选择 candidate 仍保留。

RPR 不增加 state key，因此 DS-V4 直接复用匹配的 DS-V3 merged checkpoint。

## 17. M10 Mixed Proposal Training

正式 V1/V2/V3 默认仍为 `training_proposals.source=gt_jitter`。可选 `source=mixed` 时：

- 保留 GT 和 GT-jitter。
- 额外 decode 同一次 forward 的 detached HEAL detector proposals。
- 先按 score/max_per_scene 过滤。
- 只保留 rotated IoU 大于 `positive_iou_min` 的 matched positive。
- unmatched negative 不进入 geometry refinement loss。
- proposal 与 matched target 均 detached，ROI feature 不 detach。

该路径仅在 `source=mixed` 时调用 decoder；`gt_jitter` 不付出 predicted proposal 计算成本。

## 18. Proposal Distribution Gap

默认训练 proposal 为 GT + GT-jitter，而 inference proposal 来自真实 HEAL detector，存在 distribution gap。如果 object train loss 良好但 inference refinement 无提升，应先统计真实 matched HEAL proposals 的 center/size/yaw error 分布，再调整 jitter 或启用 mixed proposal。第一反应不应是堆叠更大的网络。

## 19. Stage1: 建立两个公共空间

Stage1 m1 homogeneous 同时训练：

- 原 HEAL detection/pyramid/depth loss。
- object individual residual loss。
- object consensus loss。
- V3 quality loss。

梯度路径：

```text
object/quality loss
→ Shared Refiner / Quality Head
→ Shared Object Encoder
→ differentiable ROI sampler
→ m1 Common BEV feature
→ m1 Aligner / Backbone / Encoder
```

proposal geometry 与 quality target detached；ROI feature 不 detach。Stage1 可以从 plain HEAL checkpoint 开始，也可以从完整 lower-profile Dual-Space checkpoint warm start。任一 module group 的 partial state 会被拒绝。

## 20. Stage2: 双公共空间 backward adaptation

m2、m3、m4 仍独立训练。每个新 modality 同时适应：

- Space-I: 原 HEAL Common BEV。
- Space-II: Detail adapter；V2/V3 还包括 Context adapter。

Stage2 冻结：

- Shared Object Encoder。
- Shared Context Encoder / fusion / adaptive gate（若存在）。
- Shared Geometry Encoder / Refiner。
- Shared Quality Head（若存在）。
- 非 active modality adapters。

冻结参数无 parameter gradient，但 autograd 不切断其输入梯度。Quality loss 因此能穿过 frozen Quality Head，迫使新模态进入可被已有 quality function 正确评价的公共目标空间。

不使用 m1 teacher cosine matching，因为研究目标不是模仿某个 m1 feature vector，而是在共享几何任务与质量函数下获得可交换表示。

## 21. Merge Ownership

官方 `m2 → m3 → m4 → m1` merge order 保留，但所有 Dual-Space key 使用显式 owner 覆盖：

| Key group | Owner |
|---|---|
| shared object/geometry encoder/refiner | Stage1 m1 |
| shared context encoder | Stage1 m1 |
| shared concat fusion 或 adaptive gate | Stage1 m1 |
| shared quality head | Stage1 m1 |
| detail adapter m2/m3/m4 | 对应 Stage2 m2/m3/m4 |
| context adapter m2/m3/m4 | 对应 Stage2 m2/m3/m4 |
| RPR | 无参数、无 owner |

若某 profile 需要的 owner prefix 在对应 checkpoint 中不存在，merge fail fast，不允许静默保留错误来源。真实 Stage2 checkpoint 若包含 frozen shared backend，其 shared key 集必须与 Stage1 m1 完全一致；因此不能用 V1/V2 Stage2 权重冒充 V3，也不能输出缺少 context adapter 的不完整 merge。

## 22. Checkpoint Compatibility

| Target | Plain HEAL | V1 | V2 | V3 | 说明 |
|---|---:|---:|---:|---:|---|
| Stage1 V1 | ✓ | ✓ | × | × | 可 fresh 或完整 resume |
| Stage1 V2 | ✓ | ✓ | ✓ | × | 新 MS group 可初始化 |
| Stage1 V3 | ✓ | ✓ | ✓ | ✓ | 新 MS/Quality group 可初始化 |
| Stage2 V1 | × | ✓ | × | × | 当前 shared group 必须完整 |
| Stage2 V2 | × | × | ✓ | × | Context shared 必须完整 |
| Stage2 V3 | × | × | × | ✓ | Quality shared 必须完整 |
| Inference V1/V2/V3 | × | 对应 profile | 对应 profile | 对应 profile | 全部 expected keys 必须存在 |
| Inference V4 | × | × | × | ✓ | V4 state 与 V3 完全相同 |

任意 unexpected Dual-Space key、partial module group 或随机 Stage2/inference 初始化都会被拒绝。

## 23. 正式 Profiles

| Profile | Dual | Multi-scale | Quality | RPR | 训练需求 |
|---|---:|---:|---:|---:|---|
| HEAL | × | × | × | × | 官方 HEAL |
| DS-V1 | ✓ | × | × | × | 独立 Stage1/Stage2 |
| DS-V2 | ✓ | ✓ | × | × | 独立 Stage1/Stage2 |
| DS-V3 | ✓ | ✓ | ✓ | × | 独立 Stage1/Stage2 |
| DS-V4 | ✓ | ✓ | ✓ | ✓ | 复用 V3 merged，只改 inference policy |

配置目录：

```text
opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/
  DS_V1/{stage1_m1,stage2_m2,stage2_m3,stage2_m4,merged_infer}.yaml
  DS_V2/{stage1_m1,stage2_m2,stage2_m3,stage2_m4,merged_infer}.yaml
  DS_V3/{stage1_m1,stage2_m2,stage2_m3,stage2_m4,merged_infer}.yaml
  DS_V4/merged_infer.yaml
```

V1/V2/V3 的非 Dual-Space 训练设置由配置测试保证一致。V4 仅改变 RPR block。

## 24. 配置依赖与性能保护

validator fail-fast 规则：

- `dual_space.enabled=false` 禁止开启 MS、Quality、RPR、adaptive gate 或 mixed proposals。
- `fusion=adaptive_gate` 要求 `multi_scale.enabled=true`。
- `consensus.mode=quality_weighted` 要求 `quality.enabled=true`。
- `training_proposals.source=mixed` 要求 `predicted.enabled=true`。
- RPR 要求 `mode=inference`。
- unknown mode/fusion/consensus/proposal source 直接报错。
- 不自动修正非法配置。

关闭时保护：

- MS off: 不构造 Context adapter/encoder/fusion，不捕获 pyramid context。
- Quality off: 不构造或运行 Quality Head，不计算 rotated IoU target。
- RPR off: 不运行 per-agent detector decode。
- GT-jitter: 不 decode predicted proposals。
- report_stats off: inference 不向外返回额外诊断统计。

## 25. 可选消融

- Refiner-only。
- Dual without Stage2 Object Adaptation。
- Single-scale vs Multi-scale。
- Uniform vs Quality-weighted consensus。
- Quality ON but Multi-scale OFF。
- `concat_projection` vs `adaptive_gate`。
- embedding dim 64/128/256。
- ROI size 3/5/7。
- object loss weight。
- GT-jitter vs mixed training proposals。
- RPR OFF/ON。

每项消融应只改 YAML，并使用同一源码 commit。

## 26. 主评价指标

报告 AP@0.3、AP@0.5、AP@0.7，并分别列出 CAV1、CAV1-2、CAV1-3、CAV1-4。

边际协作收益定义为：

```text
DeltaCAV2 = AP(CAV1-2) - AP(CAV1)
DeltaCAV3 = AP(CAV1-3) - AP(CAV1-2)
DeltaCAV4 = AP(CAV1-4) - AP(CAV1-3)
```

三个 IoU threshold 都应给出边际值，不能只选择 AP@0.7。

## 27. Object 诊断指标

- proposal IoU before refinement。
- refined IoU after refinement。
- mean/median DeltaIoU。
- `P(DeltaIoU > 0)` 与 `P(DeltaIoU < 0)`。
- valid ROI ratio。
- mean ROI coverage。
- mean contributing agents/object。
- Quality calibration、mean predicted q、mean q target。
- RPR rescued proposal count、rescued TP/FP、remote recovery recall。

这些指标用于区分“refinement 无效”“少量目标被修坏”“proposal distribution gap”和“discovery failure”。

当 `dual_space.report_stats=true` 时，同一次 scene forward 返回聚合字段：`original_fused_proposals`、`rescued_remote_proposals`、`final_candidate_count`、`refined_proposal_count`、`valid_agent_object_pairs`、`mean_agents_per_object` 和 `mean_roi_coverage`。Quality profile 额外返回 `mean_quality`、`median_quality`、`low_quality_fraction`；当前诊断阈值为 `q < 0.25`。RPR profile 额外返回过滤前后、dedup 后和最终追加数量。默认 `false` 时不构造这些 inference 统计。

## 28. 计算指标

- 各 profile 新增参数量。
- GPU peak memory。
- train time/epoch。
- inference latency。
- ROI proposal count。
- Object module forward cost。
- RPR per-agent decoder 额外成本。

必须确认最大 proposal-dependent tensor 只与 `chunk_size × A × C × ROI area` 成比例，明显避免旧 Stage3 约 21 GiB 的 full-map 临时展开。

正式配置使用 detail 64 channels、context 128 channels、object embedding 128。对应参数量为：

| Group | Parameters |
|---|---:|
| V1 shared object/geometry/refiner | 138,248 |
| detail adapter（每个 m2/m3/m4） | 8,448 |
| V2 shared context encoder + concat fusion | 177,281 |
| context adapter（每个 m2/m3/m4） | 33,280 |
| V3 shared quality head | 10,497 |
| V4 RPR | 0 |

因此四模态 merged Dual-Space 参数总量为 V1 163,592、V2 440,713、V3/V4 451,210。可选 adaptive gate 取代 concat fusion 时，gate 本身为 16,513 参数。

## 29. 实验推进决策树

### Phase 0: 完整性与 GPU smoke

先完成 config parse、checkpoint load、单 batch CUDA forward/backward、AMP、peak memory 和 merged inference smoke。

### Phase 1: DS-V1

- 若 object mean DeltaIoU ≤ 0：停止，不进入 V2，先检查 box convention、transform、yaw、ROI 和 loss。
- 若 DeltaIoU > 0 但 AP 不升：检查 post-NMS refinement、score ordering、proposal distribution gap，以及少量被修坏目标。

### Phase 2: DS-V2

V1 成立后开启 Multi-scale，比较 DeltaIoU、AP、显存和 latency。若 context 无收益，检查捕获 tensor 是否确实为 pre-fusion Common BEV，以及 Context ROI physical field 是否正确。

### Phase 3: DS-V3

V2 成立后开启 Quality，重点看 DeltaCAV4、quality calibration 和低质量 agent 的负迁移是否下降。

### Phase 4: DS-V4

先统计 `remote-seen / fused-missed` 目标比例。只有比例显著时才启用 RPR；比例很低时，RPR 的 false-positive 风险可能大于收益。

## 30. 推荐实验目录

```text
opencood/logs/HEAL_DUAL_SPACE/
  ds_v1/
    stage1/m1_base/
    stage2/m2_alignto_m1/
    stage2/m3_alignto_m1/
    stage2/m4_alignto_m1/
    stage2/merged_m1m2m3m4/
    final_infer/
  ds_v2/  # same structure
  ds_v3/  # same structure
  ds_v4/  # reuse ds_v3 merged checkpoint; store RPR inference results
```

不要建立 `stage3/object_refiner/` 作为当前主方法目录。

## 31. 为什么没有最终 Stage3

主方法强调新增模态分别适应两个既有公共空间。若 merge 后再联合训练全部模态，研究命题会重新变成 closed-set joint calibration，并削弱 backward-compatible adaptation 的解释。

只有未来证明确有 merged combination calibration gap 时，才可研究 1–2 epoch tiny calibration，并只训练极少量 consensus/quality 参数。该实验不属于当前主方法。

## 32. 已知风险与诊断优先级

| 风险 | 诊断信号 | 优先级与处理 |
|---|---|---|
| Common BEV tensor 选择错误 | 单模态检测正常但 object ROI 无语义 | P0；核对 capture 在 Aligner 后、fusion 前 |
| pairwise transform 方向错误 | remote ROI 系统性镜像/平移反向 | P0；人工 translation/rotation fixture |
| rotated ROI yaw/HW 错误 | 0/90 度梯度方向不符 | P0；ROI orientation smoke |
| proposal distribution gap | train loss 好、真实 proposal DeltaIoU 差 | P1；统计真实 center/size/yaw error，调 jitter/mixed |
| Context 不是公共空间 | 某 modality Context 表现异常 | P0；只允许 shared pyramid pre-fusion level |
| Stage2 误解冻 shared module | shared key 在 m2/m3/m4 checkpoint 漂移 | P0；requires_grad/gradient ownership test |
| merge 覆盖 adapter | merged adapter 值来自 m1/错误 Stage2 | P0；prefix ownership smoke |
| Quality calibration 失真 | q 与 IoU target 不相关或饱和 | P1；calibration plot、mean q/target、分桶误差 |
| RPR false positives | rescued FP 高、AP@0.3 降 | P1；提高 score/IoU filter，必要时关闭 RPR |
| post-NMS refinement 重复框 | box 移动后重叠增加 | P1；统计 duplicate rate；当前不做第二 NMS |
| object loss 干扰 BEV | detection loss/AP@0.3 退化 | P0；降低 object weight，检查梯度比例 |
| Multi-scale 显存/延迟 | peak memory/latency 超预算 | P1；减 proposals/chunk/ROI，不扩展 full map |

## 33. 代码与测试入口

主要实现：

- `dual_space_config.py`: strict config/dependency validation。
- `dual_space_object_roi.py`: chunked rotated ROI。
- `dual_space_box_coder.py`: hwl residual、corner conversion、exact rotated IoU dispatch。
- `dual_space_proposal_sampler.py`: GT-jitter、mixed positive sampler、repository detector decoder wrapper。
- `dual_space_object.py`: adapters、encoders、multi-scale、quality、consensus、context、checkpoint 与 inference refinement。
- `dual_space_remote_proposal_rescue.py`: parameter-free RPR policy。
- `dual_space_object_loss.py`: residual + quality SmoothL1。
- `pyramid_fuse.py`: optional pre-fusion feature exposure；default return contract 不变。
- `heal_tools.py`: explicit merge ownership。
- `inference_utils.py`: RPR-before-refinement 与 optional stats return。

本地测试入口：

```text
python opencood/tools/check_dual_space_object_smoke.py
python opencood/tools/check_dual_space_multiscale_smoke.py
python opencood/tools/check_dual_space_quality_smoke.py
python opencood/tools/check_dual_space_rpr_smoke.py
python opencood/tools/check_dual_space_full_profile_smoke.py
python opencood/tools/check_dual_space_config_pack.py
```

## 34. 当前状态

当前代码开发阶段：**Full Framework implemented, local CPU functional validation complete; server validation pending**。

已完成本地验证：

- 原 DS-V1 smoke: 34/34 PASS。
- Multi-scale smoke: 17/17 PASS；本机无 CUDA，已实现的 CUDA AMP/memory case 明确 SKIP。
- Quality smoke: 21/21 PASS。
- RPR smoke: 19/19 PASS。
- mixed/config/checkpoint/merge/profile smoke: 32/32 PASS。
- 正式 YAML repository parser pack: 18/18 PASS。
- Stage2 prepare: 5/5 PASS（含 Windows symlink copy fallback）。
- Merge audit: 7/7 PASS（含错误 source、全源 missing key、shared overwrite、profile mismatch 与正式 schema）。
- Refinement diagnostics: 9/9 PASS；本机缺少 Shapely 的 IoU/inference/AP 集成 3 项明确 SKIP。
- V2/V3 synthetic forward-loss-backward。
- V3 state save/load 与 V3→V4 checkpoint reuse。
- PACT Stage3 36/36、PACT ROI 12/12、HVP-CBEA、HVP-HEAL、gradient accumulation、AMP、merge functional regression 均通过。
- 四个历史 PACT 脚本的功能断言通过，但其末尾旧任务工作区白名单护栏会因本次合法 Full Dual-Space 未提交文件而返回非零；未修改或跳过这些历史断言。

尚未验证，不得声称已通过：

- server CUDA smoke 与 compiled rotated-IoU extension。
- 使用服务器完整依赖执行真实 detector proposal decoder（本地缺少 Shapely）。
- real OPV2V DataLoader 单 batch。
- CUDA AMP/DDP 下的 Full Dual-Space 组合。
- full Stage1 m1 training。
- full Stage2 m2/m3/m4 training。
- merge 后真实 heterogeneous inference。
- 最终 AP、DeltaIoU、memory 和 latency。

## 35. 正式配置与运行契约

三种 `dual_space.mode` 不可互换：

| Mode | 职责 | 初始化/输出约束 |
|---|---|---|
| `stage1_anchor` | m1 建立 shared object/geometry backend | 仅 Stage1 可允许从 plain HEAL 初始化；训练 forward 使用 GT+jitter proposal |
| `stage2_adapt` | m2/m3/m4 分别训练本模态 adapter/aligner | `active_modality` 必须是对应 mX，shared Dual-Space backend 冻结 |
| `inference` | merged m1/m2/m3/m4 检测与 refinement | 必须加载完整 checkpoint，并返回同一次 forward 的 `dual_space_context` |

正式 merged inference 配置是：

```text
opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/DS_V1/merged_infer.yaml
opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/DS_V2/merged_infer.yaml
opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/DS_V3/merged_infer.yaml
opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/DS_V4/merged_infer.yaml
```

它们都显式使用 `mode: inference`、`allow_untrained_initialization: false` 和完整 m1/m2/m3/m4 结构。V1/V2/V3/V4 的 feature flags 分别保持原定义；训练 YAML 不应复制后手工改 mode 作为正式实验记录。

正式训练参数以 YAML 为 source of truth：`batch_size`、`accumulate_grad_batches`、`amp`、`epoches` 应保存在实验 `config.yaml`。`--accumulation-steps`、`--amp`、`--no-amp` 仅用于兼容或临时 override，不是推荐的可复现实验记录方式。

## 36. Stage2 Seed 准备

固定流程为：

```text
Stage1 唯一 net_epoch_bestval_at*.pth
  -> stage2/net_epoch1.pth
  -> stage2/m2_alignto_m1/net_epoch1.pth
  -> stage2/m3_alignto_m1/net_epoch1.pth
  -> stage2/m4_alignto_m1/net_epoch1.pth
```

使用只负责目录准备、不启动训练的工具：

```powershell
python opencood/tools/prepare_dual_space_stage2.py `
  --profile-dir opencood/hypes_yaml/HEAL_XLab_v4_DUAL_SPACE/DS_V3 `
  --stage1-dir opencood/logs/HEAL_DUAL_SPACE/ds_v3/stage1/m1_base `
  --stage2-dir opencood/logs/HEAL_DUAL_SPACE/ds_v3/stage2
```

工具要求 Stage1 best checkpoint 唯一，复制后校验 SHA256，预检三个 Stage2 config 的 mode、active modality、profile/version 和初始化策略。子目录 checkpoint 优先建立相对 symlink；Windows 权限不允许时显式回退为 byte-identical copy。目标目录非空或已有训练输出时直接拒绝，不隐式覆盖。

## 37. Merge Ownership 与审计

| Merged key group | 唯一来源 |
|---|---|
| m1 与 HEAL shared/base ownership | Stage1，遵循 `heal_tools.py` 的现有 merge contract |
| `dual_space_shared_object_encoder.*` | Stage1 |
| `dual_space_shared_geometry_encoder.*` | Stage1 |
| `dual_space_shared_object_refiner.*` | Stage1 |
| shared context/multiscale/quality groups | Stage1 |
| m2 branch/aligner、object/context adapter | Stage2 m2 |
| m3 branch/aligner、object/context adapter | Stage2 m3 |
| m4 branch/aligner、object/context adapter | Stage2 m4 |

合并完成后运行只读审计；它使用生产 `merge_dict` 与 `apply_dual_space_merge_ownership` 重建期望结果，并对 source tensor 做 exact key、shape、dtype、value 比较：

```powershell
python opencood/tools/audit_dual_space_merge.py `
  --stage1-checkpoint <stage1-best.pth> --stage1-config <stage1-config.yaml> `
  --stage2-m2-checkpoint <m2-best.pth> --stage2-m2-config <m2-config.yaml> `
  --stage2-m3-checkpoint <m3-best.pth> --stage2-m3-config <m3-config.yaml> `
  --stage2-m4-checkpoint <m4-best.pth> --stage2-m4-config <m4-config.yaml> `
  --merged-checkpoint <merged.pth> --merged-config <merged_infer.yaml> `
  --json-out <merge_audit.json>
```

Stage2 adapter 相对 seed 是否发生变化仅作为软诊断；ownership 错误、缺 key、profile 不一致或 merged tensor 不相等是硬失败。

## 38. Refinement IoU Diagnostics

Diagnostics 是 inference-only observer，默认关闭，不参与训练、loss、NMS、score、box refinement 或 checkpoint；关闭时不捕获 before/after tensor，也不新增返回字段、文件或 `state_dict` key。

```yaml
dual_space:
  diagnostics:
    enabled: false
    match_iou_min: 0.3
    thresholds: [0.3, 0.5, 0.7]
    improvement_epsilon: 1.0e-4
    save_per_object: false
```

启用时使用官方 AP 路径相同的 rotated BEV polygon IoU（不是 3D IoU）。先按 BEFORE IoU 从高到低做确定性 one-to-one matching，稳定 tie-break 为 proposal index、GT index；固定 pair 后计算：

```text
DeltaIoU = IoU_after - IoU_before
cross-up@t   = before < t 且 after >= t
cross-down@t = before >= t 且 after < t
```

JSON 包含 scene/proposal/rescued/matched counts，before/after mean IoU，mean/median DeltaIoU，positive/negative mean delta，improved/worsened/unchanged count 与 fraction，以及每个阈值的 cross-up/cross-down。普通 inference 保存 `dual_space_refinement_stats.json`；`inference_heter_in_order` 按 `use_cavN` 保存独立文件。`save_per_object: true` 时才额外写包含 scene、proposal、GT、score 和 IoU delta 的 CSV。

RPR 只比较具有 BEFORE identity 的原始 proposal；rescued proposal 单独计数。当前 refinement 显式提供 source indices metadata，该 metadata 不进入模型状态。

## 39. 硬件与复现边界

m2、m3、m4 Stage2 可在不同型号 GPU 上独立训练。Merge 只由 checkpoint 的 `state_dict` key/value 与 ownership 决定，GPU 编号和型号不属于 checkpoint 语义；不同 GPU kernel、CUDA/cuDNN 版本和算法选择仍可能造成数值级复现差异，因此每个实验必须保留实际 config、checkpoint hash 和运行环境记录。

新增工程检查入口：

```text
python opencood/tools/check_dual_space_config_pack.py
python opencood/tools/check_prepare_dual_space_stage2_smoke.py
python opencood/tools/check_dual_space_merge_audit_smoke.py
python opencood/tools/check_dual_space_diagnostics_smoke.py
```
