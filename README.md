# MedTACO

医学影像诊断问答的**工具增强信用优化**：让 Qwen3-VL-8B 学会"先写代码裁剪/放大病灶区域，再回答"，
并用 TACO 的差分答案探针（DAPR）给**每一次工具调用**分配信用，而不是只看最终答案对错。

方案全文见 [PROPOSAL.md](PROPOSAL.md)，模块接口见 [INTERFACES.md](INTERFACES.md)。

## 为什么不是普通 GRPO

| 现象 | 证据 |
|---|---|
| 工具确实有用 | Qwen3-VL-8B 给人工框做 RoI：SLAKE 66.16 → 76.15 |
| 但 outcome-only RL 学不会 | Lingshu 同规模 GRPO：均值 −0.3，感知项全负（SLAKE −0.6 / PathVQA −0.8 / OmniMedVQA −1.6）|
| 而且会"假装用工具" | DeepEyes 把返回的裁剪图随机打乱，准确率 83.3 → 83.8（没有变化）|

结论：**要的是过程信用，不是更多结果奖励**。所以走 TACO。

## 方法（相对 TACO 的三处扩展）

1. **分级前缀探针**：对 K 次调用的轨迹取 a₀…a_K，Δ_k = R(a_k) − R(a_{k−1})，
   得到逐次调用的局部信用，能区分"第一次裁对了、第二次放大反而看丢了"。
2. **医学确定性判分器**：开放答案不能用字符串匹配，但**judge 不许进训练循环**
   （成本 96 万次/epoch，且非确定性会污染 GRPO 的分母）。线上只用规则判分。
3. **多轮 OGAR 分段路由**：按 Δ_k 给 C_k ∪ T_{k+1} 打 token mask，观测 token 永远不参与 loss。

## 目录

```
medtaco/
  utils/    schema(数据结构) config io(断点续传) stats(Wilson/McNemar/DEFF)
  data/     collect+preprocess(采集/去重/phash) training_set(轨迹->训练样本)
  synth/    qgen(G1..G5) teacher(Opus5 轨迹合成) validate(防 bypassing)
  tools/    ops(crop/zoom/window/measure) sandbox(AST 白名单+超时)
  rl/       grader(确定性判分) advantage(分级探针+OGAR) loss(两趟) verl_plugin
  eval/     metrics(acc/VEG/IoU/配对检验)
scripts/    pipeline.py(入口) train_sft.sh train_rl.sh
configs/    default.yaml
tests/      单测（离线、不联网）
```

## 快速开始

```bash
PY=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/software/envs/vlm/bin/python
$PY -m pytest tests/ -q          # 全部离线单测

$PY scripts/pipeline.py --stage collect
$PY scripts/pipeline.py --stage qgen --limit 2000
$PY scripts/pipeline.py --stage synth --dry-run          # 只看成本估算
$PY scripts/pipeline.py --stage synth --mock             # 走通链路，不花钱
# 本机自带 Qwen3-VL-8B 权重时，教师用本地 8 卡跑（不花钱、可全量）：
$PY scripts/pipeline.py --stage synth --teacher local    # 见下文「本地教师」
$PY scripts/pipeline.py --stage validate --mock
$PY scripts/pipeline.py --stage export --for sft
$PY scripts/pipeline.py --stage export --for rl
```

训练（需要 GPU 节点）：

```bash
bash scripts/train_sft.sh            # 8xH800 全参；40G 卡加 --lora
bash scripts/train_rl.sh             # 需要先 clone verl
```

## 数据链路

```
图像采集 → 分层(A/B/C) → 问题合成(G1–G5) → 三道过滤 → Opus5 轨迹 → 三项防 bypassing
    → SFT → TACO 式 RL → 评测(准确率 + VEG)
```

**核心约束**：只有 A 层（有像素级标注）能监督裁剪位置；B 层只做答案监督与负样本，
**绝不伪造裁剪轨迹**。window 只在有 HU 的 CT 上可用，caliper 需要 PixelSpacing
（MIMIC-CXR-JPG 已经丢掉）。

## 状态

| 部分 | 状态 |
|---|---|
| utils / data / synth / tools / eval | 已实现 + 单测（391 项全过）|
| rl/grader, advantage, loss | 已实现 + 单测（纯算法，可离线验证）|
| rl/verl_plugin | **脚手架，未在真实 verl 上验证** |
| scripts/train_sft.sh, train_rl.sh | 已写，未在集群上跑过 |
| **真实链路 collect→qgen→synth** | **已在本机跑通**：ChestX-ray14 880 张 → 996 样本 → 4980 问题 → 本地 Qwen3-VL-8B（8×H20-143G）合成轨迹 |
| 离线链路 `pipeline.py --stage all --mock` | 已跑通：12 图 → 60 题 → 60 轨迹 → 60 条 SFT，全程不联网 |
| scripts/check_gate.py, eval_all.py | 已实现，在合成数据上验证过 |

### 本地教师（`--teacher local`）

`medtaco/synth/local_teacher.py` 把本机的 Qwen3-VL-8B-Instruct 包成**与 Anthropic
Batch API 同形**的客户端（`create / retrieve / results`），`run_batch` 一行不改就能
换成它。实现要点：

- `fork` + 每 worker 一次 `torch.cuda.set_device(idx)`，父进程不碰 CUDA；
  8 卡并行，单条请求约 17 s；
- 贪心解码（`do_sample=False`）——`validate` 要对同一问题重采样 a1/a2，非确定性会
  让「裁剪是否改变答案」的判定变成噪声；
- 输出 schema 以文本形式追加到 prompt 末尾，响应仍按 `parse_response` 解析，
  因此**离线/在线两条路径共用同一套解析与校验**。

本机为什么不用 vLLM：vLLM 0.8.5 的注册表里没有 Qwen3-VL（直接报 unsupported），
transformers 4.57 已支持。`--stage validate` 需要 OpenAI 兼容端点时用
`scripts/serve_local.py`（同一份权重，单卡串行）。

**教师输出质量的三道闸**（都是实测踩出来的，缺一条脏数据就会进 SFT）：

1. 模态不匹配的模板在 `qgen._iter_templates` 就被剔除（不能在胸片上问 CT 值）；
2. `answer_type` 由问题问法决定（`qgen._answer_type`），**不继承源样本、也不采信
   教师自报**——8B 教师在枚举上几乎恒定输出 `closed`；
3. `teacher._check_answer_field` 拒绝 `final_answer` / `probe.answer` 是
   `answer_type` 枚举值、含 `probe_no_tool` 之类字段名、或为空的情况，落到
   `data/logs/synth_failures.jsonl` 而不是训练集。

`--mock` 的数据源是 `medtaco/data/collect.py:parse_mock`：现画 12 张 1024² PNG
（9 张带框、3 张不需要工具），用途只是把 collect→qgen→synth→validate→export
整条链跑通并验证接口。**它不是训练数据**——`meta['mock']=True`，且轨迹里的答案
是 `[mock-answer-*]` 占位符。真实链路把 `data.datasets` 指向已下载的数据集即可。

尚未核实的事项见 PROPOSAL 附录 A（首要是 `claude-opus-5` 的定价与型号）。
