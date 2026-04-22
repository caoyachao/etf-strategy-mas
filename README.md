# ETF 趋势跟踪-均值回归策略 MAS 开发器

基于 **LangGraph** 的多智能体（MAS）量化策略开发系统，覆盖信号发现、策略建模、回测验证全链路。

## 架构概览

```
signal_fanout
    ├─ tech_analysis (技术信号)
    ├─ fundamental_analysis (基本面信号)
    └─ sentiment_analysis (情绪信号)
           ↓
    strategy_builder (策略代码生成)
           ↓
    backtest_runner (回测引擎 - ToolNode)
           ↓
    metrics_evaluator (指标评估)
           ↓
    refinement_router (条件路由)
    ├─ 达标 ─→ document_writer → END
    └─ 不达标 ─→ strategy_builder (迭代优化)
```

## 核心特性

- **并行信号分析**：技术面、基本面、情绪面三个 Agent 并行执行
- **结构化状态流转**：全图共享 `StrategyState`，条件路由精确控制迭代循环
- **严谨回测引擎**：支持 A 股场内 ETF，处理 T+1、等权重、止盈止损
- **自动迭代优化**：回测不达标时自动反馈给策略构建器，最多迭代 5 轮
- **完整文档输出**：策略逻辑 + 回测报告 + 参数说明的 Markdown 报告

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
│   └── backtest_engine.py   # ETF 回测引擎
├── agents/                  # Agent 配置文件（预留）
├── data/                    # ETF 历史数据缓存
└── output/                  # 运行产出（策略代码、报告、日志）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方式

### 基础运行

```bash
python strategy_mas.py \
  --task "开发一个ETF趋势跟踪-均值回归策略" \
  --output ./output
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
| `report.md` | 完整策略报告 |
| `result.md` | 执行摘要 |
| `status.json` | 结构化状态与指标 |
| `transcript.md` | 完整执行日志 |

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

## 回测引擎

`backtest/backtest_engine.py` 提供独立的回测能力：

```python
from backtest.backtest_engine import run_backtest

result = run_backtest(
    strategy_code="def generate_signals(df): ...",
    etf_pool=["510300", "510500", "159915"],
    start_date="2023-01-01",
    end_date="2025-12-31",
)
# result["sharpe"], result["max_drawdown"], result["passed"]
```

数据获取优先级：缓存 > BaoStock。

## 自定义扩展

### 增加新的信号维度

1. 在 `build_strategy_graph()` 中添加新节点
2. 在 `StrategyState` 中增加新字段
3. 在 `strategy_builder` prompt 中引用新信号

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

- 回测数据通过 BaoStock 获取真实 A 股 ETF 数据（前复权）
- 回测引擎暂未处理 ETF 分红、拆分等复权细节（已使用 BaoStock 的前复权数据）

## License

MIT
