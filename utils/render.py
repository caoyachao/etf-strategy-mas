#!/usr/bin/env python3
"""终端实时渲染器 —— 读取 execution.md 并渲染 DAG 执行状态

用法:
    # 实时监控（配合 strategy_mas.py 使用）
    tail -f output/etf-*/execution.md | python utils/render.py

    # 事后回放
    python utils/render.py < output/etf-20260423-xxxx/execution.md

    # 指定刷新间隔（秒）
    python utils/render.py --interval 0.5 < execution.md
"""

import json
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any


# DAG 节点定义
NODE_ORDER = [
    "signal_fanout",
    "tech_analysis",
    "fundamental_analysis",
    "sentiment_analysis",
    "strategy_builder",
    "coding_node",
    "backtest_runner",
    "metrics_evaluator",
    "document_writer",
]

NODE_LABELS = {
    "signal_fanout": "信号分发",
    "tech_analysis": "技术面分析",
    "fundamental_analysis": "基本面分析",
    "sentiment_analysis": "情绪面分析",
    "strategy_builder": "策略构建",
    "coding_node": "策略编码",
    "backtest_runner": "回测执行",
    "metrics_evaluator": "指标评估",
    "document_writer": "报告生成",
}

STATUS_ICONS = {
    "pending": "⬜",
    "running": "🟡",
    "success": "✅",
    "error": "❌",
}


def clear_screen():
    """清屏"""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_dag(node_states: Dict[str, str], durations: Dict[str, int], task_id: str = "") -> str:
    """渲染 DAG 状态为 ASCII 流程图"""
    lines = []
    lines.append("")
    lines.append(f"  ╔═══════════════════════════════════════════════════════════════╗")
    lines.append(f"  ║  ETF 策略 MAS 执行监控 {'':<33}║")
    if task_id:
        lines.append(f"  ║  Task: {task_id:<49}║")
    lines.append(f"  ╚═══════════════════════════════════════════════════════════════╝")
    lines.append("")

    # 并行分析阶段
    lines.append("  ┌──────────────────────────────────────────┐")
    lines.append("  │  阶段一：并行信号分析                     │")
    lines.append("  └──────────────────────────────────────────┘")

    for node in ["signal_fanout", "tech_analysis", "fundamental_analysis", "sentiment_analysis"]:
        label = NODE_LABELS.get(node, node)
        status = node_states.get(node, "pending")
        icon = STATUS_ICONS.get(status, "⬜")
        dur = durations.get(node, 0)
        dur_str = f"({dur}ms)" if dur > 0 else ""
        lines.append(f"     {icon} {label:<20} {dur_str:>10}")

    lines.append("")
    lines.append("  ┌──────────────────────────────────────────┐")
    lines.append("  │  阶段二：策略构建与编码                   │")
    lines.append("  └──────────────────────────────────────────┘")

    for node in ["strategy_builder", "coding_node"]:
        label = NODE_LABELS.get(node, node)
        status = node_states.get(node, "pending")
        icon = STATUS_ICONS.get(status, "⬜")
        dur = durations.get(node, 0)
        dur_str = f"({dur}ms)" if dur > 0 else ""
        lines.append(f"     {icon} {label:<20} {dur_str:>10}")

    lines.append("")
    lines.append("  ┌──────────────────────────────────────────┐")
    lines.append("  │  阶段三：回测与评估                       │")
    lines.append("  └──────────────────────────────────────────┘")

    for node in ["backtest_runner", "metrics_evaluator"]:
        label = NODE_LABELS.get(node, node)
        status = node_states.get(node, "pending")
        icon = STATUS_ICONS.get(status, "⬜")
        dur = durations.get(node, 0)
        dur_str = f"({dur}ms)" if dur > 0 else ""
        lines.append(f"     {icon} {label:<20} {dur_str:>10}")

    lines.append("")
    lines.append("  ┌──────────────────────────────────────────┐")
    lines.append("  │  阶段四：文档输出                         │")
    lines.append("  └──────────────────────────────────────────┘")

    for node in ["document_writer"]:
        label = NODE_LABELS.get(node, node)
        status = node_states.get(node, "pending")
        icon = STATUS_ICONS.get(status, "⬜")
        dur = durations.get(node, 0)
        dur_str = f"({dur}ms)" if dur > 0 else ""
        lines.append(f"     {icon} {label:<20} {dur_str:>10}")

    lines.append("")

    # 统计
    total = len(NODE_ORDER)
    completed = sum(1 for s in node_states.values() if s == "success")
    running = sum(1 for s in node_states.values() if s == "running")
    errors = sum(1 for s in node_states.values() if s == "error")

    lines.append(f"  进度: {completed}/{total} 完成 | {running} 运行中 | {errors} 失败")

    if completed == total:
        lines.append("  🎉 所有节点执行完毕！")
    elif errors > 0:
        lines.append("  ⚠️  部分节点执行失败，请检查日志")

    lines.append("")
    return "\n".join(lines)


def process_line(line: str, node_states: Dict[str, str], durations: Dict[str, int]) -> bool:
    """处理单条 JSONL 记录，更新状态。返回是否需要重绘"""
    line = line.strip()
    if not line:
        return False

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False

    node = record.get("node", "")
    event = record.get("event", "")

    if node not in NODE_ORDER:
        return False

    if event == "start":
        node_states[node] = "running"
        return True
    elif event == "end":
        node_states[node] = "success"
        durations[node] = record.get("duration_ms", 0)
        return True
    elif event == "error":
        node_states[node] = "error"
        durations[node] = record.get("duration_ms", 0)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="LangGraph 执行监控渲染器")
    parser.add_argument("--interval", type=float, default=0.3, help="刷新间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只渲染一次后退出（用于事后回放）")
    args = parser.parse_args()

    node_states: Dict[str, str] = {n: "pending" for n in NODE_ORDER}
    durations: Dict[str, int] = {}
    task_id = ""

    # 初始渲染
    clear_screen()
    print(render_dag(node_states, durations, task_id))
    sys.stdout.flush()

    if args.once:
        # 一次性读取所有行
        for line in sys.stdin:
            process_line(line, node_states, durations)
            if not task_id:
                try:
                    r = json.loads(line.strip())
                    task_id = r.get("task_id", "")
                except:
                    pass
        clear_screen()
        print(render_dag(node_states, durations, task_id))
        return

    # 实时模式：逐行读取
    while True:
        line = sys.stdin.readline()
        if not line:
            # stdin 关闭，退出
            break

        changed = process_line(line, node_states, durations)
        if changed:
            if not task_id:
                try:
                    r = json.loads(line.strip())
                    task_id = r.get("task_id", "")
                except:
                    pass
            clear_screen()
            print(render_dag(node_states, durations, task_id))
            sys.stdout.flush()

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
