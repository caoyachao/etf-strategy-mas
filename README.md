# ETF 趋势跟踪-均值回归策略 MAS 开发器

基于 **LangGraph** 的多智能体（MAS）量化策略开发系统，覆盖数据获取、信号分析、策略建模、回测验证、文档输出全链路。

## 架构概览

```
signal_fanout
    ↓
data_loader (统一数据获取：K线 + 指标 + 基本面)
    ├─ tech_analysis (技术信号)
    ├─ fundamental_analysis (基本面信号)
    └─ sentiment_analysis (情绪信号)
           ↓
    strategy_builder (策略配置生成)
           ↓
    coding_node (策略代码编译)
           ↓
    backtest_runner (回测引擎 - ToolNode)
           ↓
    metrics_evaluator (指标评估)
           ↓
    refinement_router (条件路由)
    ├─ 达标 ─→ document_writer → graph_end
    └─ 不达标 ─→ strategy_builder (迭代优化，最多5轮)
```

## 核心特性

- **数据与分析解耦**：统一 `data_loader` 节点从 baostock 获取真实数据并计算技术指标，LLM 只做分析不做数据获取，消除幻觉
- **并行信号分析**：技术面、基本面、情绪面三个 Agent 并行执行，基于同一数据源
- **完整 LLM 日志**：每轮 LLM 对话追加记录到 `llm_logs/{role}.md`，支持多轮迭代追溯
- **实时节点追踪**：`execution.md` 记录每个节点的执行事件，`status.json` 实时更新进度
- **终端 DAG 渲染**：`utils/render.py` 实时可视化节点执行状态
- **最优策略选择**：迭代上限未达标时，从 `strategy_history` 中用 Min-Max 归一化评分选出综合最优策略
- **后台运行支持**：通过 `nohup` 后台执行，随时通过 `status.json` 查询进度
- **严谨回测引擎**：支持 A 股场内 ETF，处理 T+1、等权重、止盈止损

## 回测通过标准

| 指标 | 阈值 |
|------|------|
| 夏普比率 | ≥ 2.0 |
| 最大回撤 | ≤ 5% |
| 超额收益（相对沪深300） | ≥ 5% |

## 项目结构

```
.
├── README.md
├── requirements.txt
├── strategy_mas.py          # 主入口：LangGraph MAS 工作流
├── backtest/
│   ├── __init__.py
│   ├── backtest_engine.py   # ETF 回测引擎（支持外部数据注入）
│   └── data_fetcher.py      # 统一数据获取：拉数据 + 算指标 + 基本面
├── agents/
│   └── openclaw_llm.py      # LLM 调用封装
├── utils/
│   ├── llm_logger.py        # LLM 对话日志（追加模式）
│   ├── tracing.py            # LangChain Callback 节点追踪
│   ├── render.py             # 终端 DAG 实时渲染
│   └── output_manager.py     # 输出目录管理
├── data/                    # ETF 历史数据缓存（.gitignore 排除）
└── output/                  # 运行产出（.gitignore 排除）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方式

### 基础运行（前台）

```bash
python strategy_mas.py \
  --task "开发一个ETF趋势跟踪-均值回归策略" \
  --output ./output
```

### 后台运行（推荐）

```bash
cd /path/to/etf-strategy-mas
nohup python strategy_mas.py \
  --task "开发一个ETF趋势跟踪-均值回归策略" \
  > output/run.md 2>&1 &
```

后台运行后，随时查看进度：

```bash
# 查看最新任务状态
cat output/etf-*/status.json

# 实时监控日志
tail -f output/run.md

# 实时渲染 DAG 进度
tail -f output/etf-*/execution.md | python utils/render.py
```

### 自定义阈值

```bash
python strategy_mas.py \
  --task "开发高夏普ETF策略" \
  --output ./output \
  --sharpe-threshold 2.5 \
  --drawdown-threshold 0.03 \
  --excess-threshold 0.08 \
  --max-iterations 5
```

### 产出文件

运行后在 `output/{task-id}/` 下生成：

| 文件 | 内容 |
|------|------|
| `strategy.py` | 可执行的策略 Python 代码 |
| `report.md` | 完整策略报告（Markdown） |
| `result.md` | 执行摘要 |
| `status.json` | 实时状态与指标（后台查询用） |
| `execution.md` | 结构化执行日志（每行 JSON，支持 render.py 渲染） |
| `llm_logs/` | LLM 对话日志（按角色分文件，追加模式） |

## 接入真实 LLM

接入 OpenClaw：

编辑 `.env`：

```bash
USE_OPENCLAW=true
# 可选：自定义角色到 Agent ID 的映射（默认全部用 main）
# OPENCLAW_ROLE_MAP=tech_analyst:analyst1,strategy_dev:coder
# 可选：单轮超时（秒）
# OPENCLAW_TIMEOUT=120
```

代码自动通过 `openclaw agent --agent <id> --message <prompt> --json` 调用你的 OpenClaw 智能体，**复用已有 provider 配置，无需额外 API Key**。

默认所有角色都调用 `main` agent，可通过 `OPENCLAW_ROLE_MAP` 为不同角色分配不同的 OpenClaw Agent。

各 Agent 的角色 prompt 已内嵌在对应节点函数中（`tech_analysis`、`fundamental_analysis` 等），可直接使用。

## 数据获取与分析解耦

`backtest/data_fetcher.py` 提供统一数据获取：

```python
from backtest.data_fetcher import load_all_data, get_data_summary

# 拉取所有 ETF 数据（含技术指标）
data = load_all_data(
    etf_pool=["510300", "510500"],
    start_date="2023-01-01",
    end_date="2025-12-31",
)

# 生成 LLM 分析用的数据摘要
summary = get_data_summary(data, fundamental={}, n_recent=5)
```

数据流：
1. `data_loader` 节点统一拉取原始数据 + 计算技术指标
2. `tech_analysis` / `fundamental_analysis` / `sentiment_analysis` 消费摘要文本
3. `backtest_runner` 复用 `data_loader` 已加载的数据，不再重复拉取

## 节点追踪系统

`utils/tracing.py` 基于 LangChain Callback 实现：

- 每个节点进入/退出时自动记录
- 写入 `execution.md`（每行一个 JSON 事件）
- 实时更新 `status.json`（状态 + 进度 + 指标）
- 支持 `__node__` 显式标识（100% 准确）

查看 DAG 可视化：

```bash
tail -f output/etf-*/execution.md | python utils/render.py
```

## 回测引擎

`backtest/backtest_engine.py` 提供独立的回测能力：

```python
from backtest.backtest_engine import run_backtest

result = run_backtest(
    strategy_code="def generate_signals(df): ...",
    etf_pool=["510300", "510500", "159915"],
    start_date="2023-01-01",
    end_date="2025-12-31",
    data=preloaded_data,  # 可选：传入预加载数据，避免重复拉取
)
# result["sharpe"], result["max_drawdown"], result["passed"]
```

数据获取优先级：缓存 > BaoStock。

## 迭代优化与最优策略选择

回测不达标时，系统自动反馈给策略构建器迭代优化（最多 5 轮）。

迭代上限未达标时，从 `strategy_history` 中用 **Min-Max 归一化评分** 选出最优策略：

```
score = sharpe_norm × 3 + drawdown_norm × 2 + excess_norm × 1
```

- `sharpe_norm`：该轮 sharpe 在所有轮次中的归一化值（越高越好）
- `drawdown_norm`：回撤越低，归一化值越高
- `excess_norm`：超额收益越高，归一化值越高

确保三个指标在同一量级（0~1），避免数量级差异导致的权重失衡。

## 自定义扩展

### 增加新的信号维度

1. 在 `build_strategy_graph()` 中添加新节点
2. 在 `StrategyState` 中增加新字段
3. 在 `strategy_builder` prompt 中引用新信号
4. 在 `data_fetcher.py` 中添加对应的数据摘要生成函数

### 修改 ETF 池

编辑 `strategy_mas.py` 中的 `DEFAULT_ETF_POOL`：

```python
DEFAULT_ETF_POOL = [
    "510300",   # 沪深300
    "510500",   # 中证500
    # 添加更多 ETF 代码
]
```

### 调整策略类型

修改 `strategy_builder` 节点中的 prompt，改变 `generate_signals()` 的生成要求。

## 已知限制

- 回测数据通过 BaoStock 获取真实 A 股 ETF 数据（前复权），当前数据覆盖期约 3.5 个月（2026-01-05 ~ 2026-04-22）
- 回测引擎暂未处理 ETF 分红、拆分等复权细节（已使用 BaoStock 的前复权数据）
- 策略参数（RSI 阈值、均线周期等）在短数据周期上较难达到高夏普、低回撤的严格阈值

## License

MIT
