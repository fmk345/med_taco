# 数据样例

从全量数据里各抽**一条** SFT / RL 样本，把训练侧会用到的三要素（图片、问题、答案）
连同真实记录一起放进仓库，克隆下来即可直接读，不需要下载任何数据集。

## 文件

| 文件 | 内容 |
|---|---|
| `images/chestx_ray14_00018762_001.png` | 原图。ChestX-ray14 `00018762_001.png`，1024×1024 灰度 |
| `images/crop_0.png` | 第 1 次 `crop` 工具返回的病灶区域裁图 |
| `images/crop_1.png` | 第 2 次 `crop` 工具返回的裁图 |
| `question.json` | `questions.jsonl` 里对应的那一行：原始问题、金标框 `boxes`、`answer_type`、来源与过滤元数据 |
| `answer.txt` | 金标答案 `no` |
| `sft_example.jsonl` | **SFT 样本**，一行，字段与 `data/train/sft.jsonl` 完全一致 |
| `rl_example.jsonl` | **RL 样本**，一行，字段与 `data/train/rl.jsonl` 完全一致 |

两份 jsonl 里所有图片路径都已改写成**仓库相对路径**（`examples/images/...`），
全量文件里是 `/mnt/dolphinfs/...` 的绝对路径——这是唯一的差别。
尾部多了一个 `_example` 字段，只是给人看的注释（图片路径 / 问题 / 金标答案 / 对应问题记录的位置），
训练脚本读的是 `messages`、`images`、`probe_scores` 等原有字段，加这一个键不影响解析；
不想要的话删掉即可。

样本 id：`traj-chestx_ray14:5b07627b32a7dae4#G1-zh-05`

## 这条样本是什么

ChestX-ray14 一张胸片，G1（框锚定）模板生成的中文是非题：*"图中标记框内的 Mass 与周围组织相比，
边界是否清晰？"*，`answer_type=closed`，金标 `no`（教师作答回填，见 `scripts/backfill_answers.py`）。

轨迹是 2 次工具调用的标准形态：

```
user      [原图] + 问题
assistant 假设：原图中未见明确肿块，边界不清。      ← T1，第一次工具调用前的思考
assistant <tool_call>crop(x0=100, y0=100, x1=300, y1=300)</tool_call>
user      [crop_0] + 【工具返回】
assistant <tool_call>crop(x0=500, y0=100, x1=700, y1=300)</tool_call>
user      [crop_1] + 【工具返回】
assistant <answer>no</answer>
```

## SFT 与 RL 记录的区别

两者 `messages` 完全相同，差别只在 RL 多出信用分配所需的字段：

| 字段 | SFT | RL | 含义 |
|---|:--:|:--:|---|
| `messages` / `images` | ✓ | ✓ | 多轮对话；`images` 是该条样本用到的全部图片，按出现顺序 |
| `meta.n_calls` / `meta.sample_id` | ✓ | — | 工具调用次数、回指 `questions.jsonl` 的 id |
| `sample_id`（顶层） | — | ✓ | 同上，RL 侧提到顶层 |
| `probe_scores` | — | ✓ | `a₀…a_K`，每次前缀探针的判分（`-1/0/1`），长度必须 `= n_calls + 1` |
| `delta_k` | — | ✓ | `Δ_k = a_k − a_{k−1}`，长度 `= n_calls` |
| `delta_total` | — | ✓ | `a_K − a_0` |
| `segments_char` | — | ✓ | `t1` / `calls[k] = (obs_s, obs_e, t2_s, t2_e)`，字符偏移；OGAR 据此给 token 打 mask，观测段永远不参与 loss |

> ⚠️ 本例的 `probe_scores` 是 **`[0]`**，长度 1，而 `n_calls = 2`——
> 这正是当前数据集已知的缺陷：教师合成阶段只写了占位探针，2009/2009 条轨迹全部如此。
> 重采（`medtaco/synth/probes.py`）尚未在本批数据上跑过，所以这里保留原样，
> **不要**把它当成正确的 `a₀…a_K`。重采后应是长度 3 的数组、`delta_k` 长度 2。

## 怎么读

```python
import json
rec = json.loads(open("examples/rl_example.jsonl").readline())
print(rec["messages"][0]["content"][1]["text"])   # 问题
print(rec["images"])                              # 3 张图，相对仓库根
print(rec["probe_scores"], rec["delta_k"])        # (待重采)
```

```bash
# 校验：两份样例的图片路径都能解析、并且与 messages 里引用的图片一致
python -c "
import json, os
for n in ('sft','rl'):
    d=json.load(open(f'examples/{n}_example.jsonl'))
    assert all(os.path.exists(p) for p in d['images']), n
    print(n, 'ok', d['images'])
"
```
