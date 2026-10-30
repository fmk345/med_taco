# 模块接口契约（所有模块必须遵守）

只依赖 `medtaco.utils.*` 与本文件列出的签名。跨模块调用一律用下面这些函数，
不要自己发明第二套数据结构。配置通过 `medtaco.utils.config.load_config` 传 `Cfg`。

## medtaco/utils/schema.py（已实现，勿改）
`Box(x0,y0,x1,y1,label)` / `Sample` / `ToolCall` / `ToolResult` / `Step` / `Probe` / `Trajectory` / `Verdict`
全部有 `to_json()`；除 `Probe`/`Verdict` 外有 `from_json(d)`。

## medtaco/utils/io.py（已实现，勿改）
`iter_jsonl(path)`, `seen_ids(path)`, `JsonlWriter(path, id_key="id")`, `atomic_write_json`, `shard_by_id`, `map_parallel(items, fn, workers, desc)`

## medtaco/utils/stats.py（已实现，勿改）
`wilson_ci`, `mcnemar`, `mcnemar_power_n`, `deff`, `effective_n`, `bootstrap_ci`, `paired_bootstrap`, `judge_attenuation`, `summarize_acc`

---

## medtaco/tools/ops.py
```python
def load_image(path: str) -> np.ndarray            # 返回 HxWx3 uint8
def crop(img: np.ndarray, box: Box, pad_ratio: float = 0.0) -> np.ndarray
def zoom(img: np.ndarray, box: Box, out_side: int = 1024) -> np.ndarray   # 裁剪 + 上采样到 out_side 长边（不超原图像素数上限，超了不放大，返回实际倍率）
def window(img: np.ndarray, center: float, width: float, meta: dict) -> np.ndarray
    # meta 必须含 rescale_slope/intercept 与 photometric='MONOCHROME2' 才能还原 HU；
    # 含 window_center/window_width 可选。缺 HU 信息时 raise ToolUnavailable
def measure(box: Box, meta: dict, mode: str = "long") -> float
    # meta 含 pixel_spacing=(sx,sy) 才能算 mm；缺则 raise ToolUnavailable
class ToolUnavailable(Exception): ...
def execute(call: ToolCall, ctx: "ToolContext") -> ToolResult
```
`ToolContext` 是本模块的 dataclass：`image: np.ndarray, meta: dict, out_dir: str, obs_max_side: int, jpeg_quality: int`。
`execute` 负责：参数校验 → 调用原语 → 把返回图写盘（避免内存里堆大数组）→ 填 `ToolResult.meta['crop_box']`。
所有越界框必须 `box.clip(w,h)`，退化框（宽或高 < 8px）返回 `ok=False`。

## medtaco/tools/sandbox.py
```python
def run_code(code: str, ctx, cfg) -> ToolResult      # 执行模型写的 Python，返回 ToolResult
class SandboxError(Exception): ...
def static_check(code: str, cfg) -> None             # AST 检查，非法 raise SandboxError
```
AST 白名单：只允许 `Assign/AugAssign/Expr/For/If/FunctionDef/Return/BinOp/Compare/Call/Subscript/Attribute(白名单)/Name`
禁止：`Import/ImportFrom`（除了 cfg 里 allow_imports 的模块，通过预注入命名空间暴露）、`While`、`With`、`Try`、
双下划线属性、`eval/exec/open/__import__/globals/locals/getattr/setattr/delattr/compile/input`。
运行环境：预注入 `image`(np.ndarray)、`crop/zoom/window/measure/load_image`、`math/statistics`、常量。
必须设置 timeout（cfg.rl.sandbox.timeout_s）。`backend=gvisor` 时用 `runsc` 拉起子进程；不可用时**明确降级并打 warning**，不要假装隔离。

## medtaco/rl/grader.py —— 确定性判分器（PROPOSAL §7.2.1，线上唯一允许的实现）
```python
class DeterministicGrader:
    def __init__(self, cfg, synonyms: dict[str, list[str]] | None = None)
    def score(self, sample: Sample, pred: str) -> int      # ∈ {-1,0,1}
    def gradable(self, sample: Sample) -> bool             # 该样本能否进 RL
    def parse(self, raw: str) -> tuple[str, bool]          # 从 <answer>..</answer> 抽取，返回 (answer, ok)
```
规则：
- `closed`：归一化后精确匹配；yes/no 与 affirm/deny 词表等价；否则 0。
- `short`：归一化（小写、去冠词/标点、单复数、单位统一 mm/cm、数字规范化）后，双向包含任一方向成立即 1。
  但**否定词必须一致**：`no effusion` 不得匹配 `effusion`（先剥否定前缀再比，两侧否定极性必须相同）。
- `measure`：解析数字 + 单位，相对误差 ≤10% → 1，≤25% → 0 分档记为 0 但 `meta` 记 half_credit，>25% → -1? **不做负分**：返回 0。
- `long`：`gradable()` 返回 False（线上不判），但提供 `score_offline_judge()` 明确抛 NotImplementedError。
- 解析失败（`parse` 的 ok=False）由调用方记 0 并不进 Δ；`score` 本身对空串返回 -1。
必须提供 `to_dict()/from_dict()` 以保存同义词表和配置，保证**同输入同输出**（写单测证明两次调用一致）。

## medtaco/rl/advantage.py —— 分级探针 + OGAR（PROPOSAL §7.1/§7.3）
```python
@dataclass
class TurnMasks:
    token_mask: list[int]        # 长度 = 轨迹 token 数
    call_indices: list[int]      # 每次调用在 token 序列中的位置

def staged_deltas(probe_scores: list[int]) -> list[int]     # a_0..a_K -> Δ_1..Δ_K
def ogar_masks(probe_scores: list[int], segments: list[Segment], mode: str = "paper") -> TurnMasks
@dataclass
class Segment:
    t1: tuple[int, int]              # 第一段思考 token 区间 [start, end)
    calls: list[tuple[int, int, int, int]]  # 每次: (obs_start, obs_end, t2_start, t2_end)
def group_advantage(values: list[float], std_eps: float, policy: str) -> list[float] | None
    # 返回 None 表示该组零方差，应跳过（policy='skip_group'）
def build_loss_masks(...) -> dict   # 供两趟 loss 使用：{'loss_mask_1':..., 'loss_mask_2':..., 'adv_1':..., 'adv_2':...}
```
规则严格按 PROPOSAL §7.3：
- Δ_total = 探针末 - 探针首；`m[t]=1` for T1 iff Δ_total>=0；对每个 k，`C_k ∪ T_{k+1}` 当 Δ_k≠0；否则 0。
- `g = 1[Δ_total>=0]`，作用于 process 通道（A2）。
- 观测 token（`obs_*` 区间）永远 mask。
- `zero_var_policy='skip_group'`：组内所有值相同则返回 None；`'eps_floor'`：std 置为 max(std, std_eps)。
- 解析失败的探针：调用方应剔除该轨迹；本模块提供 `validate_probes()` 检查长度与单调性。

## medtaco/rl/loss.py —— 两趟 policy loss（防 agg_loss 改尺度）
```python
def two_pass_policy_loss(logprobs, old_logprobs, advantages, loss_masks, clip_eps, alpha_1, alpha_2) -> dict
```
返回 `{'loss':..., 'loss_1':..., 'loss_2':..., 'pg_clipfrac':..., 'n_tokens_1':..., 'n_tokens_2':...}`。
两个通道**各自按自己的 mask 求 token 均值**再乘 α，绝不能用拼接后的总 mask 归一化。
必须支持 torch 与纯 python 列表两种输入（测试环境未必有 torch）。

## medtaco/data/collect.py / preprocess.py
- `collect_dataset(name, cfg) -> Iterator[Sample]`：按 cfg.data.datasets 的 tier 走不同解析器；
  未实现的数据集返回空并打印 `[skip]`，不要抛。
- `scan_manifest(cfg) -> list[dict]`：产出 `{dataset,image,box_count,tier,license}`。
- `phash_dedup(paths, hamming, tile_hash, grid) -> dict[str, str]`：返回 dup -> kept 映射。
- `preprocess_image(src, dst, cfg) -> dict`：统一到 PNG/JPEG + 写 sidecar json（原始尺寸、spacing、window）。
- `audit_resolution(samples) -> dict`：可 zoom / 不可 zoom 统计。

## medtaco/synth/qgen.py
```python
def generate(samples: list[Sample], cfg, model_fn=None) -> list[Sample]   # 产出问题（answer 可为空，留待教师填）
def filter_questions(samples, cfg, base_model_fn=None, agree_fn=None) -> list[Sample]
```
G1..G5 每个一个函数 `g1_box_anchored(...)` 等，可单测。模板要中文/英文双语字典。
`model_fn(prompt, images) -> str` 是注入点，测试时传 mock，**不要硬编码任何 SDK 调用**。

## medtaco/synth/teacher.py
```python
def build_request(sample: Sample, cfg) -> dict            # 一条轨迹一次请求（3 个 image block + json_schema）
def estimate_cost(n_samples: int, cfg, calls_per_traj: float) -> dict
def run_batch(samples, cfg, client=None) -> list[Trajectory]   # client 可注入 mock
def parse_response(resp: dict, sample: Sample) -> Trajectory
```
schema 拍平、全字段 required、additionalProperties=false；**禁止 message prefill**。

## medtaco/synth/validate.py —— 三项防 bypassing（PROPOSAL §5.3）
```python
def premise_perturbation(traj, sample, tool_ctx, ask_fn, cfg) -> Verdict
def no_tool_ablation(traj, sample, ask_fn, cfg) -> Verdict
def tool_necessity(traj, sample, cfg) -> Verdict
def validate_all(traj, sample, ..., cfg) -> Verdict
```
面积匹配随机裁剪：面积比在 `1±area_match_tol` 内、且与 gold box 的 IoU ≤ 0.1；替换后答案**必须变化**
才证明裁剪是"承重"的；`min_answer_change` 次试验里至少 1 次变化，否则 `passed=False, reason="decorative_crop"`。

## medtaco/eval/metrics.py
```python
def iou(a: Box, b: Box) -> float
def veg(acc_normal: float, acc_corrupted: float) -> float
def oracle_crop_ceiling(...) -> dict
def paired_report(model_a: list[bool], model_b: list[bool], n_boot, icc, m) -> dict
    # 返回 acc/ci/mcnemar p/有效样本量
def per_source_report(records: list[dict], source_key="dataset") -> dict   # OmniMedVQA 必须分源
```

## medtaco/rl/verl_plugin/（verl 侧集成，无法在本容器验证）
- `adv_est.py`：`@register_adv_est("taco_staged")` 的 advantage estimator，读 rollout 里附带的
  `probe_scores` 与 `segments`，调用 `advantage.py`。
- `rollout.py`：`TacoAgentLoop` 伪代码 + 明确标注哪些方法是 verl 需要 monkeypatch 的。
- `README.md`：把"需要在 verl 侧对齐的 5 个点"逐条写清，并说明本容器无法验证。
**这部分必须诚实标注为"未在真实 verl 上验证"。**

## CONTRACT：轨迹 token 编号
`Segment.t1 = (s,e)`、`calls[k] = (obs_s, obs_e, t2_s, t2_e)`，
obs 区间是返回图/观测token（必须 mask），t2 是该轮模型后续思考+调用。
`token_mask` 与 `input_ids` 等长。若某段不适用填 `(-1,-1)`。

## 通用要求
- 中文注释，类型标注齐全，不引入除 numpy/PIL 之外的新依赖（torch 可选导入）。
- 每个模块配 `tests/test_<模块>.py`，必须能在 `pytest` 下通过且**不联网、不读大文件**。
- 不要修改 `medtaco/utils/{schema,config,io,stats}.py` 和 `configs/default.yaml`。
- 所有随机性用显式 seed。
