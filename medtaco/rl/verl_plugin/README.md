# verl 侧集成说明（**未在真实 verl 上验证**）

> 本目录全部代码是**脚手架**。开发容器里没有 verl、没有 GPU，也没有真实 tokenizer，
> 所以下面每一条都只经过离线单测（`tests/test_verl_plugin.py`），
> **没有一行在真实 verl 训练循环里跑过**。上机前请按每条的"怎么验证"逐条过一遍。

涉及的 verl 参考位置（按 verl main 分支的公开结构写出，具体行号以你的版本为准）：

- `verl/trainer/ppo/adv_estimator.py` — `AdvantageEstimator` 枚举 + `register_adv_est` + `compute_advantage`
- `verl/trainer/ppo/ray_trainer.py` — `compute_advantage(...)` 的调用点、group 构造、`agg_loss`
- `verl/trainer/ppo/core_algos.py` — `agg_loss` / `compute_policy_loss`
- `verl/experimental/agent_loop/` （或旧版 `verl/workers/rollout/...`）— `AgentLoopBase`、`tool_agent_loop.py`
- `verl/utils/torch_functional.py` / `continuous_token.py::align_response_metadata` — 观测 token 掩码

---

## 1. raw `input_ids` fork（探针的调用方式）

**为什么**：分级探针要在"同一前缀"上做 greedy 续写，每 k 次调用一次。
`continue_final_message`（vLLM 的风格）在 verl 里不存在，也没法配合 SGLang 的前缀缓存。

**在哪改**

- `verl/experimental/agent_loop/*.py`：新增一个 `TacoAgentLoop(AgentLoopBase)`，
  把 `rollout.py::TacoAgentLoop` 的四个注入点换成真实实现：
  - `generate` → `sglang_engine.generate(input_ids=..., sampling_params=..., rid=...)`
  - `tokenize` → processor 的 tokenizer（**不要**再走 chat template 重拼）
  - `run_tool` → `medtaco.tools.ops.execute`
  - `grade` → `medtaco.rl.grader.DeterministicGrader.parse/score`

**改成什么**

```python
# rollout.py 里的 build_probe_ids 已经是这个语义：
probe_ids = acc_ids + tokenize("</think>\n<answer>")      # acc_ids 是**原始** input_ids
out = engine.generate(
    input_ids=probe_ids,
    sampling_params=SamplingParams(max_new_tokens=32, temperature=0.0),
    rid=base_rid,          # 与主轨迹同一个 SessionParams.rid → radix 前缀缓存直接命中
)
```

要点三条：
1. 直接传 **raw input_ids**，不要传 messages（SGLang 侧 messages 会重新套模板，前缀就变了）；
2. `rid` 必须复用主轨迹的，否则 radix cache hit 率掉到 0，探针成本回到 0.2–0.8 s/次；
3. 探针的 32 token **不写回** `response_ids`（它只是读数），只记 `probe_scores`。

**怎么验证**

```python
# 起一次 1 卡的 dry-run，对同一条轨迹连续两次 build_probe_ids，断言前缀完全相同：
assert probe_a[:len(acc)] == acc and probe_b[:len(acc)] == acc
# 再看 SGLang 的 metrics：prefix_cache_hit_rate 应随 k 递增，而不是恒定 0
```

单测里对应 `test_build_probe_ids_appends_delimiter_only`。

---

## 2. 观测 token mask（两趟 loss 都要求）

**为什么**：返回图 / 沙箱输出是**环境给的**，不是模型生成的，不能进 policy gradient；
算进去等于让策略为环境 token 负责，梯度直接被污染。

**在哪改**

- `verl/utils/continuous_token.py::align_response_metadata`（verl main 已原生做这件事）
  或 `verl/trainer/ppo/core_algos.py::compute_policy_loss` 之前的 mask 组装处。
- 我们的入口是 `adv_est.py::TacoStagedAdvantageEstimator`，它把 `loss_mask_1` / `loss_mask_2`
  写进 `data.batch`，**由 core_algos 消费**，不要自己另造一个 mask 传进去。

**改成什么**

```
response_mask = verl 原生 response_mask                    # 已有
obs_mask      = 1 - 观测区间                                 # rollout 产出，与 response 等长
loss_mask_1   = response_mask * obs_mask                    # A1 通道
loss_mask_2   = response_mask * obs_mask * m[t] * g         # A2 通道，m[t] 来自 OGAR
```

`m[t]`、`g` 由 `medtaco/rl/advantage.py::ogar_masks` 给出，`adv_est.py` 负责对齐宽度
（`_fit_mask`：超长截断、不足右侧补 0）。

**怎么验证**

```python
# 造一条"图正中间出现"的轨迹，断言观测 token 位置上 advantages == 0：
assert (adv_1[0, obs_s:obs_e] == 0).all()
# 再做一次"mask 掉观测后 loss 尺度不变"的检查：把同一 batch 的观测换成另一张图，
# 只要长度相同，loss 必须逐位一致（观测不参与计算）。
```

单测里对应 `test_obs_mask_zeroes_observation_span`。

---

## 3. tool 返回图要作为**独立 user message** 重新注入

**为什么**：这是本工程最隐蔽的坑。多数 chat template 对 `role=tool` 只渲染文本，
image block 会被静默丢掉——模型看不见返回图，但 loss 照算，训练出来的是"假装看见"。
**必须在 tool message 之外，单独追加一条 user 消息带图。**

**在哪改**

- `medtaco/rl/verl_plugin/rollout.py::build_observation_messages`（顺序在这里定死）
- 真实 verl 侧：`tool_agent_loop.py` 里替换 `ToolResponse` 的处理，
  以及 `verl/experimental/agent_loop/agent_loop.py` 里把 `multi_modal_data` 传给 SGLang 的地方。

**改成什么**

顺序必须是：

```
assistant(tool_calls=[...])                 # 回显模型的选择
tool(content="工具读数/错误文本")            # 只有文本，绝不放图
user([{"type":"image","image":<返回图>}, {"type":"text","text":"..."}])   # ← 图在这里
```

落盘 token 时，第 3 条消息的图像 token 全部计入 obs span（`obs_mask=0`）。

**怎么验证**

```python
msgs = loop.build_observation_messages(call, result)
assert [m["role"] for m in msgs] == ["assistant", "tool", "user"]
assert any(b["type"] == "image" for b in msgs[-1]["content"])
assert isinstance(msgs[1]["content"], str)          # tool 消息里不能有图
```

**上机必做的对照实验**：同一条轨迹跑两次，一次走"独立 user message"、一次走"塞进 tool message"，
对比模型在第二轮里的答案是否引用返回图（例如 crop 之后能否读出病灶）。
如果两次答案一样，说明你的模板把图吃掉了 —— 这正是本坑。

---

## 4. 两趟 loss（不许让 `agg_loss` 改尺度）

**为什么**：`agg_loss` 按**求和后的 loss mask** 归一化。如果把 A1、A2 拼成一个 mask 一起算，
A2 通道的分母会被 A1 的 token 数摊薄，**α₂ 的实际尺度悄悄变了**（PROPOSAL §7.3）。
TACO 的敏感性曲线说 α₂ 在 0.05–0.5 之间不敏感，但那是"正确尺度"下的结论。

**在哪改**

- `verl/trainer/ppo/core_algos.py::compute_policy_loss` 的调用点，改成调
  `medtaco/rl/loss.py::two_pass_policy_loss`（同伴模块），参数：
  `(logprobs, old_logprobs, advantages, loss_masks, clip_eps, alpha_1, alpha_2)`

**改成什么**

```python
# 必须两次调用，各自用**自己的** mask 求 token 均值：
loss_1 = agg_loss(pg_loss_1, loss_mask_1)     # shape [bs, L], mask_1[:, t] ∈ {0,1}
loss_2 = agg_loss(pg_loss_2, loss_mask_2)
loss   = alpha_1 * loss_1 + alpha_2 * loss_2
# 绝不允许：
# loss = agg_loss(pg_loss, loss_mask_1 + loss_mask_2)   ← α₂ 尺度被改掉
```

同时从 `adv_est.py` 拿 `adv_1` / `adv_2`（已乘过 mask），不要再乘一次。

**怎么验证**

```python
# 尺度不变性：把 A2 的 mask 整体乘 0，loss 必须严格等于 alpha_1 * loss_1
loss_a2_off = two_pass_policy_loss(..., loss_masks={"loss_mask_2": zeros})
assert math.isclose(loss_a2_off["loss"], alpha_1 * loss_a2_off["loss_1"], rel_tol=1e-6)
# 再验证 n_tokens_1 / n_tokens_2 是各自的和，而不是总和：
assert out["n_tokens_2"] == int(loss_mask_2.sum())
```

单测里对应 `test_two_pass_does_not_mix_masks`（用损失模块的 mock 检查调用次数与参数形状）。

---

## 5. 零方差组跳过

**为什么**：Δ 只有 5 个离散取值，G=8 时一组里常常全 0。组内 std=0 →
advantage 分母为 0，要么 NaN，要么放大噪声。**原论文只有离线 D.5 过滤（§7.3），
在线跳过是我们的工程决策，必须如实标注。**

**在哪改**

- `adv_est.py::TacoStagedAdvantageEstimator._group_norm`：调 `advantage.group_advantage`，
  返回 `None` 即组内零方差 → 该组所有样本 `skipped=True`、advantage 置 0，
  写进 `non_tensor_batch["taco_skip_uid"]`。
- **真正剔除必须在 trainer 侧做**：`verl/trainer/ppo/ray_trainer.py` 在
  `compute_advantage(...)` 之后、`update_actor` 之前，按 `taco_skip_uid` 过滤 batch 元素。
  只打标不剔除，等于给模型喂一堆 0 梯度样本，白白占 batch。

**改成什么**

```python
# ray_trainer.py，compute_advantage 之后
skip = batch.non_tensor_batch.get("taco_skip_uid")
if skip is not None and skip.any():
    keep = ~skip.astype(bool)
    batch = batch[keep]          # DataProto 支持按 index 取子集
    metrics["taco/skipped_groups"] = int(skip.sum())
# 若剔除后 batch 为空，直接跳过这次 update（返回 metrics，不进 optimizer.step）
```

保守做法（不想改 batch 形状）：保留 `skip_group` 打标 + `eps_floor` 兜底
（`std_eps=1e-6`，见 `configs/default.yaml`），但要在论文里写明是 ours。

**怎么验证**

```python
# 造一组 probe_scores 全相同（Δ_total 全 0）的 G=8，断言：
assert data.non_tensor_batch["taco_skip_uid"].all()
assert not np.isnan(data.batch["advantages"]).any()   # 不能出 NaN
assert (data.batch["loss_mask_1"] == 0).all()          # 整组不进 loss
# 再混入一组正常值，断言只跳过该组：
assert 0 < data.non_tensor_batch["taco_skip_uid"].sum() < 8
```

单测里对应 `test_zero_var_group_is_skipped` 与 `test_mixed_groups_skip_only_bad_one`。

---

## 上机顺序建议

1. 先只接 **rollout**（不改 loss），用 `temperature=1.0`、`collect_staged=false` 跑 100 条，
   肉眼检查消息顺序、obs span、`input_ids` 前缀是否一致；
2. 打开分级探针（`rl.probe.collect_staged=true`），只记 `probe_scores`、不改 advantage，
   统计探针耗时占比（应 <10%）与解析失败率（应 <5%）；
3. 接入 `adv_est`，**先单测** `compute_advantage` 的输出形状与 NaN 检查，
   再开小 lr（1e-7）跑 50 step，盯 `taco/skipped_groups` 与 `pg_clipfrac`；
4. 最后才换 `two_pass_policy_loss`，并用上面第 4 条的尺度不变性测试卡一道。

---

## 明确没做的部分

- 没有任何真实 verl 版本适配（`register_adv_est` 在不同版本接受 str / 枚举，见 `adv_est.py` 的 `TODO(verl 对齐)`）；
- 没有接 SGLang 的 `multi_modal_data`，image block 格式是 messages 语义的占位；
- 没有真实 tokenizer 的 image token 数（`RolloutCfg.image_token_count=256` 是占位，
  真实值必须由 processor 给出，否则 obs span 宽度会错）；
- 没有覆盖率、没有性能数据。**不要把本目录的任何数字当实验结果引用。**
