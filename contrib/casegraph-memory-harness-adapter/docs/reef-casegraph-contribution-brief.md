# Reef 首轮 CaseGraph memory–harness 贡献草案

核验日期：2026-09-19（美国东部时间）  
目标仓库：[Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef)  
本地状态：projectless；工作区没有 Reef checkout，因此没有修改上游源代码、没有推送、没有开 PR，也没有向外部发送消息。

## 仓库核验

当前 `main` 页面显示仓库公开，许可证文件是 Apache License 2.0；`pyproject.toml` 同时声明 `license = "Apache-2.0"`，要求 Python `>=3.12`。当前页面列出 `.github/`、`recipes/`、`reef/`、`tests/`、`CONTRIBUTING.md`、`LICENSE`、`README.md`、`pyproject.toml`。README 的闭环是 serve → observe → grow → commit：

- observe 由 `reef/storage/records.py` 和 `reef/train/processors/` 承担；
- grow 由 `reef/recipe/`、`reef/train/` 承担；
- commit 由 `reef/train/evaluation/`、`reef/artifact/`、`reef/surface/` 承担。

已读到的关键契约如下：

- [`CONTRIBUTING.md`](https://github.com/Human-Agent-Society/reef/blob/main/CONTRIBUTING.md)：方向或范围不确定时先开 issue；issue/PR/fixture 不得含 credentials、private data、provider token 或 prompt transcript；常规检查是 `pre-commit run --all-files pytest tests/`；涉及 persisted format、wire contract、trust boundary、recipe extension contract 或新的 `reef` 顶层包时先写 RFC。
- [`pyproject.toml`](https://github.com/Human-Agent-Society/reef/blob/main/pyproject.toml)：Python 3.12、pytest marker `unit/integration/acceptance/sandbox`、测试入口 `tests/`、coverage floor 80。
- [`.github/workflows/ci.yml`](https://github.com/Human-Agent-Society/reef/blob/main/.github/workflows/ci.yml)：CI 在 3.12 上运行 source/sandbox 两套测试，常规命令为 `python -m pytest tests -m "not sandbox" -n 8 --dist loadfile`；sandbox 单独串行运行。
- [`reef/storage/records.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef/storage/records.py)：只负责 append/replay/retire/audit/compaction；明确不负责 scenario lifecycle、commit、artifact publication 或 training。
- [`reef/train/evaluation/evaluators.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef/train/evaluation/evaluators.py)：candidate `evaluate()` 与 `decide()` 插件，含 `RegressionCheckMixin`，可把 retained 分数作为保留/拒绝门。
- [`reef/artifact/artifact.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef/artifact/artifact.py) 与 [`release_chain.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef/artifact/release_chain.py)：artifact 支持 stage/publish/discard，release 有 parent 链，候选可在不移动 current/checkpoint head 的情况下准备和回滚。
- [`reef/surface/adapter.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef/surface/adapter.py)：已有按 `(scenario, runtime_load_id)` 可逆生成 adapter 名称的契约，不应在本草案中重做。
- [`recipes/`](https://github.com/Human-Agent-Society/reef/tree/main/recipes)：现有方法代码放在 cookbook；实验性工作有 `recipes/beta/`，示例通常包含 fixture、`run.py`、`README.md` 和 stack 配置。

测试入口和贡献入口齐全，但当前没有本地 checkout，无法安全地做 upstream import、服务启动、Git LFS artifact 或 sandbox 验证。

## Issue 对齐

- [#522](https://github.com/Human-Agent-Society/reef/issues/522) 当前开放，标题为“Memory as harness state”。RFC 提议 `memory` node、读 surface、admission/redaction、ablation，并明确 recipe 决定 entry schema、reader、write cadence。
- [#514](https://github.com/Human-Agent-Society/reef/issues/514) 当前开放，标题为“Define retained harness evaluation”。草案 PR [#515](https://github.com/Human-Agent-Society/reef/pull/515) 仍待验收；RFC 要求显式 release、原子 episode 结果、invalid/error/unrun 分类和 task-paired retained 对照。
- [#532](https://github.com/Human-Agent-Society/reef/issues/532) 当前开放且标记 `status: needs-triage`，标题为“Separate when a scenario trains from what it trains on”。`TrainingTrigger` RFC 将“何时训练”与 processor 的“能否组 batch/训练什么”分开。

因此本首轮草案只在 recipe/example/test 层提供可审查输入和回归，不新增 `reef` 顶层包，不改变 wire/persisted format，不改生产模型服务。是否把 `memory` node 或 retained evaluator 接到核心由维护者在 RFC 中决定。

## 最小贡献设计

实现位于 [`work/reef_casegraph_adapter.py`](../work/reef_casegraph_adapter.py)，是独立 prototype，不复制 Reef 代码。它包含四个可迁移的契约：

1. `CaseEvent`：固定字段为 `case_id/source_id/event_type/text_ref/observed_at/valid_from/valid_to/provenance/confidence/feedback_type/outcome/update_surface/artifact_version/cost`。所有 ID 和文本引用都以 `syn-`/`synthetic://` 开头，provenance 记录 generator、seed 和 fixture 来源。
2. `SyntheticCaseEventGenerator(seed=20260919)`：用固定 seed 生成 8 个合成 case、每个 case 一条 observation 和一条 corrective feedback；时间按两天间隔排列，避免反馈事件跨 split 边界。
3. `CaseGraphAdapter`：observation、事实错误、检索遗漏写入 episodic memory node；执行策略失败和验证器错误只增加 procedural harness 计数。重复事件按 canonical hash 幂等，错误不会静默覆盖事实来源。
4. `VersionedArtifact` 与 `select_or_rollback()`：candidate 带 parent artifact、state digest 和 consumed event IDs；retained 分数跌破 baseline 时保留 current，候选保持未选中。该边界对应 Reef 的 stage/publish/discard 与 regression-check 语义。

fixture 已导出为 [`outputs/reef-caseevent-fixture.json`](./reef-caseevent-fixture.json)，其中包含事件、split case IDs 和固定 `event_digest`。它只含合成文本引用，不含 PHI、真实病例材料、内部 prompt 或客户标识。

每个事件同时带有 `event_id = sha256(canonical_event_json)`；`CaseEvent.from_dict()` 和 `deserialize()` 会拒绝被篡改的 ID、无效 provenance 或错误的 artifact version。合成 retained 指标已写入 [`outputs/retained_evaluation_report.json`](./retained_evaluation_report.json)，显式记录 replay/adapt/retained/drift 的 quality、traceability、review time、compute/human cost 和 negative transfer。

## 评估协议

fixture 分成四个互斥集合：

| split | case IDs | 用途 |
| --- | --- | --- |
| replay | `syn-case-00`…`03` | 验证历史轨迹可重放及 digest 稳定 |
| adapt | `syn-case-04`、`05` | 应用反馈并生成 candidate artifact |
| retained | `syn-case-06` | 独立保留集；调参和适配不读取 |
| drift | `syn-case-07` | 后续分布漂移回归 |

四类反馈归因固定为 `fact_error`、`retrieval_omission`、`execution_strategy_failure`、`verifier_error`。报告时必须分别记录：

- provenance 完整率和 evidence traceability；
- temporal validity 违规数、case overlap 数、retained set 是否被改变；
- verified quality、失败类型分布、负迁移；
- reviewer/人工介入秒数、compute 毫秒、可计费调用数；
- replay/adapt/retained/drift 四段结果和 candidate 回滚原因。

这里的 adapter 只生成状态和 artifact identity，不调用 LLM、数据库、HTTP 服务或生产模型。独立 retained 评估在上游接入时应复用 `reef.train.evaluation` 的 candidate plugin；不要把保留集反馈写回 `RecordStore` 的训练可见读取。

## 拟向上游迁移的文件和边界

待维护者在 #522/#514/#532 讨论后，再考虑以下最小文件集合：

- `recipes/beta/casegraph/`：方法适配和合成 fixture，不新增 `reef` 顶层包；
- `recipes/beta/casegraph/examples/synthetic_casegraph/fixture.json`、`run.py`、`README.md`：可运行 deterministic replay；
- `tests/test_casegraph_synthetic.py`：unit、temporal/case-disjoint、feedback attribution、duplicate-event regression；
- `tests/reef_service/test_casegraph_memory_harness.py`：在有 checkout 和依赖时做 integration，核验 recipe surface、candidate artifact 和 retained gate；
- `docs/user-guide/recipes/casegraph.rst`：解释 provenance、temporal validity、feedback credit assignment、cost/negative transfer 与 rollback。

第一 PR 不应修改生产模型服务、训练后端、`reef/storage/records.py` 的持久化格式、`reef/surface/adapter.py` 的命名契约，或在 retained 集上调参。若必须新增 memory node kind、read route 或 persisted schema，应先按 CONTRIBUTING 的 RFC 规则把设计放回 #522。

## 本地验证

已运行：

```text
uv run --with pytest python -m pytest -q test_reef_casegraph_adapter.py
11 passed in 0.04s
```

测试文件是 [`work/test_reef_casegraph_adapter.py`](../work/test_reef_casegraph_adapter.py)，并用 `unit`、`integration`、`regression` marker 覆盖：固定 seed、序列化 round-trip、invalid provenance、tampered event ID、temporal leakage、artifact-version mismatch、split 不泄漏、四类反馈归因、replay/candidate identity、retained regression rollback、duplicate event 幂等。

上游 checkout 可用后的验证命令：

```text
python -m pytest tests/test_casegraph_synthetic.py -m unit
python -m pytest tests/reef_service/test_casegraph_memory_harness.py -m integration
python -m pytest tests/test_casegraph_synthetic.py -m regression
pre-commit run --all-files
python -m pytest tests/
```

本机没有 `pytest` 系统安装，验证使用了 `uv run --with pytest` 的临时环境；本机 Python 是 3.14，而 Reef 当前要求 3.12，所以没有宣称通过 Reef 的完整 CI、sandbox、Postgres、package 或 coverage gate。

## 阻塞项和下一步

阻塞项是：没有安全可修改的上游 checkout；#522 尚是 RFC；#514 的 retained evaluator 草案仍待验收；#532 仍需 triage；核心 memory node、read route、TrainingTrigger 与 persisted schema 的所有权尚未由维护者确认。

下一步应先把本 brief 和合成 fixture 作为 issue 讨论材料，请维护者确认：adapter 是否放在 `recipes/beta/`、retained manifest 是否沿用 #514 的 `reef-release-evaluation/2` 方向、feedback type 命名是否接受、以及 memory-only candidate 是否走 #522 的 selection 语义。得到明确边界后，再在独立 checkout 中迁移上述最小文件并运行 Reef 的 Python 3.12 CI 命令；在此之前保持本地草案，不推送、不建 PR。
