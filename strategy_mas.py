"""
ETF 趋势跟踪-均值回归策略 MAS 开发器
基于 LangGraph 的多智能体协作系统

架构：
  信号并行分析（技术/基本面/情绪） → 策略构建 → 回测（ToolNode）
  → 指标评估 → 条件路由（达标/不达标循环） → 文档生成

使用方式：
  python strategy_mas.py --task "开发一个ETF趋势跟踪策略" --output ./output/
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import operator
import pandas as pd
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# 回测引擎
from backtest.backtest_engine import run_backtest, ETFBacktestEngine

# LLM 桥接层 & 输出管理
from agents.openclaw_llm import get_llm
from utils.output_manager import OutputManager

# 加载 .env 文件（如果存在）
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ETF_POOL = [
    "510300",   # 沪深300ETF
    "510500",   # 中证500ETF
    "512100",   # 中证1000ETF
    "159915",   # 创业板ETF
    "588000",   # 科创50ETF
    "512690",   # 酒ETF
    "512800",   # 银行ETF
    "512880",   # 证券ETF
    "515030",   # 科技ETF
    "515790",   # 光伏ETF
]

PASS_THRESHOLD = {
    "sharpe": 2.0,
    "max_drawdown": 0.05,
    "excess_return": 0.05,
}

MAX_ITERATIONS = 5

BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"

OUTPUT_DIR = Path("./output")


# ─────────────────────────────────────────────────────────────────────────────
# State 定义
# ─────────────────────────────────────────────────────────────────────────────

class StrategyState(TypedDict):
    task: str
    etf_pool: List[str]
    tech_signals: str
    fundamental_signals: str
    sentiment_signals: str
    strategy_code: str
    backtest_result: Dict[str, Any]
    metrics: Dict[str, Any]
    iteration: int
    max_iterations: int
    pass_threshold: Dict[str, float]
    status: str
    messages: Annotated[List[Dict], operator.add]
    feedback: str
    final_document: str


# ─────────────────────────────────────────────────────────────────────────────
# 节点函数
# ─────────────────────────────────────────────────────────────────────────────

def signal_fanout(state: StrategyState) -> Dict:
    """初始化状态，准备并行信号分析"""
    return {
        "messages": [{"node": "signal_fanout", "role": "init", "content": f"Task: {state['task']}"}],
        "iteration": 0,
        "status": "running",
    }


def tech_analysis(state: StrategyState) -> Dict:
    """技术面信号分析 Agent"""
    llm = get_llm("tech_analyst")
    etf_list = ", ".join(state["etf_pool"])

    prompt = (
        f"你是一位资深的技术分析专家。请对以下ETF进行技术面分析，"
        f"输出每只ETF的技术信号评分（-1到1之间）。\n\n"
        f"ETF池: {etf_list}\n"
        f"分析维度: MA均线、RSI、MACD、布林带\n\n"
        f"任务: {state['task']}\n"
    )

    if state.get("feedback"):
        prompt += f"\n上一轮回测反馈（如需优化参考）: {state['feedback']}\n"

    response = llm.call(prompt)

    return {
        "tech_signals": response,
        "messages": [{"node": "tech_analysis", "role": "analyst", "content": response}],
    }


def fundamental_analysis(state: StrategyState) -> Dict:
    """基本面信号分析 Agent"""
    llm = get_llm("fundamental_analyst")
    etf_list = ", ".join(state["etf_pool"])

    prompt = (
        f"你是一位基本面分析专家。请对以下ETF进行基本面分析，"
        f"输出每只ETF的基本面信号评分（-1到1之间）。\n\n"
        f"ETF池: {etf_list}\n"
        f"分析维度: 净值、溢价率、规模变化、估值分位\n\n"
        f"任务: {state['task']}\n"
    )

    if state.get("feedback"):
        prompt += f"\n上一轮回测反馈（如需优化参考）: {state['feedback']}\n"

    response = llm.call(prompt)

    return {
        "fundamental_signals": response,
        "messages": [{"node": "fundamental_analysis", "role": "analyst", "content": response}],
    }


def sentiment_analysis(state: StrategyState) -> Dict:
    """情绪面信号分析 Agent"""
    llm = get_llm("sentiment_analyst")
    etf_list = ", ".join(state["etf_pool"])

    prompt = (
        f"你是一位市场情绪分析专家。请对以下ETF进行情绪面分析，"
        f"输出每只ETF的情绪信号评分（-1到1之间）。\n\n"
        f"ETF池: {etf_list}\n"
        f"分析维度: 资金流向、成交量异动、市场情绪热度\n\n"
        f"任务: {state['task']}\n"
    )

    if state.get("feedback"):
        prompt += f"\n上一轮回测反馈（如需优化参考）: {state['feedback']}\n"

    response = llm.call(prompt)

    return {
        "sentiment_signals": response,
        "messages": [{"node": "sentiment_analysis", "role": "analyst", "content": response}],
    }


def strategy_builder(state: StrategyState) -> Dict:
    """策略构建 Agent：汇总三信号，生成策略代码"""
    llm = get_llm("strategy_dev")

    prompt = (
        f"你是一位量化策略开发工程师。请基于以下三个维度的分析信号，"
        f"编写一个完整的 Python 策略函数 `generate_signals(df)`。\n\n"
        f"## 技术面信号\n{state['tech_signals']}\n\n"
        f"## 基本面信号\n{state['fundamental_signals']}\n\n"
        f"## 情绪面信号\n{state['sentiment_signals']}\n\n"
        f"## 要求\n"
        f"1. 函数签名必须是: `def generate_signals(df):`\n"
        f"2. df 包含以下列: 'open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'ma60', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct'\n"
        f"3. 返回一个 pandas Series，值为 -1(看空), 0(中性), 1(看多)\n"
        f"4. 策略逻辑应结合趋势跟踪（价格在均线上方）和均值回归（RSI超卖+布林带低位）\n"
        f"5. 策略应适配 ETF 特性（T+1、等权重、最多同时持有3只）\n"
        f"6. 输出必须包含完整的 Python 代码块\n"
    )

    if state.get("iteration", 0) > 0:
        prompt += (
            f"\n## 优化反馈（第 {state['iteration']} 轮迭代）\n"
            f"{state['feedback']}\n\n"
            f"请根据反馈优化策略参数和逻辑。"
        )

    response = llm.call(prompt)

    # 提取代码块
    code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        strategy_code = code_match.group(1).strip()
    else:
        # 尝试提取 def generate_signals
        func_match = re.search(r"(def generate_signals\(.*?:.*?)(?=\n##|\n\n# |$)", response, re.DOTALL)
        strategy_code = func_match.group(1).strip() if func_match else response

    return {
        "strategy_code": strategy_code,
        "messages": [{"node": "strategy_builder", "role": "developer", "content": response}],
        "iteration": state.get("iteration", 0) + 1,
    }


def backtest_runner(state: StrategyState) -> Dict:
    """回测执行器（Tool Node 封装）"""
    strategy_code = state["strategy_code"]
    etf_pool = state["etf_pool"]

    print(
        f"[backtest_runner] Running backtest, iteration={state.get('iteration', 0)}",
        file=sys.stderr,
    )

    result = run_backtest(
        strategy_code=strategy_code,
        etf_pool=etf_pool,
        start_date=BACKTEST_START,
        end_date=BACKTEST_END,
        pass_threshold=state["pass_threshold"],
    )

    return {
        "backtest_result": result,
        "metrics": {
            "sharpe": result.get("sharpe", 0),
            "max_drawdown": result.get("max_drawdown", 1),
            "excess_return": result.get("excess_return", -1),
            "annual_return": result.get("annual_return", 0),
            "trade_count": result.get("trade_count", 0),
            "win_rate": result.get("win_rate", 0),
        },
        "messages": [
            {
                "node": "backtest_runner",
                "role": "engine",
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }
        ],
    }


def metrics_evaluator(state: StrategyState) -> Dict:
    """指标评估 + 反馈生成"""
    result = state["backtest_result"]
    threshold = state["pass_threshold"]

    metrics = state["metrics"]
    passed = result.get("passed", False)

    feedback_parts = []
    if not passed:
        if metrics["sharpe"] < threshold["sharpe"]:
            feedback_parts.append(
                f"夏普比率 {metrics['sharpe']:.3f} 低于阈值 {threshold['sharpe']:.3f}。"
                f"建议：降低策略波动率，增加趋势过滤条件，或收紧入场标准。"
            )
        if metrics["max_drawdown"] > threshold["max_drawdown"]:
            feedback_parts.append(
                f"最大回撤 {metrics['max_drawdown']:.2%} 超过阈值 {threshold['max_drawdown']:.2%}。"
                f"建议：加强止损逻辑（如移动止损），或降低单次仓位。"
            )
        if metrics["excess_return"] < threshold["excess_return"]:
            feedback_parts.append(
                f"超额收益 {metrics['excess_return']:.2%} 低于阈值 {threshold['excess_return']:.2%}。"
                f"建议：优化选股信号权重，增加基本面或情绪面筛选。"
            )

    feedback = "\n".join(feedback_parts) if feedback_parts else "所有指标通过阈值。"

    return {
        "feedback": feedback,
        "messages": [
            {
                "node": "metrics_evaluator",
                "role": "evaluator",
                "content": f"Passed: {passed}\nMetrics: {json.dumps(metrics)}\nFeedback: {feedback}",
            }
        ],
    }


def refinement_router(state: StrategyState) -> str:
    """条件路由：达标/迭代上限 → END，否则 → strategy_builder"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    passed = state.get("backtest_result", {}).get("passed", False)

    print(
        f"[refinement_router] iteration={iteration}/{max_iter}, passed={passed}",
        file=sys.stderr,
    )

    if passed:
        print("[refinement_router] ✓ Metrics passed → END", file=sys.stderr)
        return "document_writer"

    if iteration >= max_iter:
        print(f"[refinement_router] ✗ Max iterations reached ({max_iter}) → END", file=sys.stderr)
        return "document_writer"

    print("[refinement_router] → strategy_builder (refinement)", file=sys.stderr)
    return "strategy_builder"


def document_writer(state: StrategyState) -> Dict:
    """文档生成 Agent：整合所有信息输出最终报告"""
    llm = get_llm("document_writer")

    metrics = state["metrics"]
    result = state["backtest_result"]
    passed = result.get("passed", False)
    iteration = state.get("iteration", 0)

    prompt = (
        f"你是一位量化策略文档撰写专家。请基于以下信息，生成一份完整的策略报告。\n\n"
        f"## 策略代码\n```python\n{state['strategy_code']}\n```\n\n"
        f"## 技术面分析\n{state['tech_signals']}\n\n"
        f"## 基本面分析\n{state['fundamental_signals']}\n\n"
        f"## 情绪面分析\n{state['sentiment_signals']}\n\n"
        f"## 回测结果\n"
        f"- 夏普比率: {metrics['sharpe']:.3f} (阈值 ≥ {state['pass_threshold']['sharpe']})\n"
        f"- 最大回撤: {metrics['max_drawdown']:.2%} (阈值 ≤ {state['pass_threshold']['max_drawdown']:.0%})\n"
        f"- 超额收益: {metrics['excess_return']:.2%} (阈值 ≥ {state['pass_threshold']['excess_return']:.0%})\n"
        f"- 年化收益: {metrics.get('annual_return', 0):.2%}\n"
        f"- 交易次数: {metrics['trade_count']}\n"
        f"- 胜率: {metrics['win_rate']:.1%}\n"
        f"- 回测区间: {BACKTEST_START} ~ {BACKTEST_END}\n"
        f"- 迭代轮次: {iteration}\n"
        f"- 通过状态: {'✓ 通过' if passed else '✗ 未通过'}\n\n"
        f"请输出一份专业的 Markdown 格式策略报告，包含策略概述、选股池、交易逻辑、"
        f"回测结果、参数说明、风险提示等章节。"
    )

    response = llm.call(prompt)

    return {
        "final_document": response,
        "status": "complete" if passed else "failed",
        "messages": [{"node": "document_writer", "role": "writer", "content": response}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 图构建器
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy_graph() -> StateGraph:
    """构建完整的策略开发 StateGraph"""
    graph = StateGraph(StrategyState)

    # 添加节点
    graph.add_node("signal_fanout", signal_fanout)
    graph.add_node("tech_analysis", tech_analysis)
    graph.add_node("fundamental_analysis", fundamental_analysis)
    graph.add_node("sentiment_analysis", sentiment_analysis)
    graph.add_node("strategy_builder", strategy_builder)
    graph.add_node("backtest_runner", backtest_runner)
    graph.add_node("metrics_evaluator", metrics_evaluator)
    graph.add_node("document_writer", document_writer)

    # 入口点
    graph.set_entry_point("signal_fanout")

    # signal_fanout → 并行三个分析
    graph.add_edge("signal_fanout", "tech_analysis")
    graph.add_edge("signal_fanout", "fundamental_analysis")
    graph.add_edge("signal_fanout", "sentiment_analysis")

    # 三个分析 → strategy_builder（LangGraph 会自动等三个都完成）
    graph.add_edge("tech_analysis", "strategy_builder")
    graph.add_edge("fundamental_analysis", "strategy_builder")
    graph.add_edge("sentiment_analysis", "strategy_builder")

    # strategy_builder → backtest_runner
    graph.add_edge("strategy_builder", "backtest_runner")

    # backtest_runner → metrics_evaluator
    graph.add_edge("backtest_runner", "metrics_evaluator")

    # metrics_evaluator → refinement_router（条件路由）
    graph.add_conditional_edges(
        "metrics_evaluator",
        refinement_router,
        {
            "strategy_builder": "strategy_builder",  # 不达标，回退优化
            "document_writer": "document_writer",    # 达标或达上限，生成文档
        },
    )

    # document_writer → END
    graph.add_edge("document_writer", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETF 策略 MAS 开发器")
    parser.add_argument("--task", required=True, help="策略开发任务描述")
    parser.add_argument("--output", default="./output", help="输出目录")
    parser.add_argument("--task-id", default=None, help="任务ID（默认自动生成）")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help="最大迭代次数")
    parser.add_argument("--sharpe-threshold", type=float, default=PASS_THRESHOLD["sharpe"])
    parser.add_argument("--drawdown-threshold", type=float, default=PASS_THRESHOLD["max_drawdown"])
    parser.add_argument("--excess-threshold", type=float, default=PASS_THRESHOLD["excess_return"])
    args = parser.parse_args()

    task_id = args.task_id or f"etf-{datetime.now().strftime('%Y%m%d')}-{str(hash(args.task))[-4:]}"
    output_dir = Path(args.output) / task_id
    out_mgr = OutputManager(output_dir, task_id)

    initial_state: StrategyState = {
        "task": args.task,
        "etf_pool": DEFAULT_ETF_POOL,
        "tech_signals": "",
        "fundamental_signals": "",
        "sentiment_signals": "",
        "strategy_code": "",
        "backtest_result": {},
        "metrics": {},
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "pass_threshold": {
            "sharpe": args.sharpe_threshold,
            "max_drawdown": args.drawdown_threshold,
            "excess_return": args.excess_threshold,
        },
        "status": "running",
        "messages": [],
        "feedback": "",
        "final_document": "",
    }

    print(f"[main] Starting task: {args.task}", file=sys.stderr)
    print(f"[main] Task ID: {task_id}", file=sys.stderr)
    print(f"[main] Output: {output_dir}", file=sys.stderr)

    try:
        graph = build_strategy_graph()
        final_state = graph.invoke(initial_state)

        # 保存产出
        out_mgr.save_strategy(final_state["strategy_code"])
        out_mgr.save_report(final_state["final_document"])

        # 输出结果摘要
        metrics = final_state["metrics"]
        result_summary = (
            f"## 执行完成\n\n"
            f"- 迭代次数: {final_state['iteration']}\n"
            f"- 最终状态: {final_state['status']}\n"
            f"- 夏普比率: {metrics.get('sharpe', 0):.3f}\n"
            f"- 最大回撤: {metrics.get('max_drawdown', 0):.2%}\n"
            f"- 超额收益: {metrics.get('excess_return', 0):.2%}\n"
            f"- 交易次数: {metrics.get('trade_count', 0)}\n"
            f"- 胜率: {metrics.get('win_rate', 0):.1%}\n\n"
            f"产出文件:\n"
            f"- 策略代码: {output_dir / 'strategy.py'}\n"
            f"- 策略报告: {output_dir / 'report.md'}\n"
            f"- 执行日志: {output_dir / 'transcript.md'}\n"
            f"- 状态文件: {output_dir / 'status.json'}\n"
        )

        out_mgr.complete(result_summary, final_state["iteration"], metrics)
        print(result_summary)

    except Exception as exc:
        tb_str = traceback.format_exc()
        out_mgr.error(exc, initial_state.get("iteration", 0), tb_str)
        print(f"[main] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
