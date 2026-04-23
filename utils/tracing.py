"""节点追踪与日志系统

基于 LangChain Callback 的监控方案：
- 在每个 LangGraph 节点进入/退出时自动记录
- 写入结构化 JSONL 日志文件
- 支持终端实时渲染（配合 render.py 使用）
- **实时写入 status.json，供外部查询进度**

用法（在 strategy_mas.py 中）:
    from utils.tracing import NodeTraceCallback
    callback = NodeTraceCallback(out_dir=Path("./output"))
    graph.invoke(state, config={"callbacks": [callback]})
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler


class NodeTraceCallback(BaseCallbackHandler):
    """LangGraph 节点追踪回调

    通过 run_id 精确匹配 start/end 事件，避免 LangGraph 内部 wrapper 导致的
    serialized 为空的问题。
    """

    # DAG 节点顺序（用于终端渲染时的定位）
    NODE_ORDER = [
        "signal_fanout",
        "data_loader",
        "tech_analysis",
        "fundamental_analysis",
        "sentiment_analysis",
        "strategy_builder",
        "coding_node",
        "backtest_runner",
        "metrics_evaluator",
        "refinement_router",
        "document_writer",
        "graph_end",
    ]

    NODE_LABELS = {
        "signal_fanout": "信号分发",
        "data_loader": "数据加载",
        "tech_analysis": "技术面分析",
        "fundamental_analysis": "基本面分析",
        "sentiment_analysis": "情绪面分析",
        "strategy_builder": "策略构建",
        "coding_node": "策略编码",
        "backtest_runner": "回测执行",
        "metrics_evaluator": "指标评估",
        "refinement_router": "条件路由",
        "document_writer": "报告生成",
        "graph_end": "执行结束",
    }

    # 根据输出字段反推节点名
    # 注意：顺序和唯一性很重要，避免 signal_fanout（也输出 iteration）被误判
    _OUTPUT_TO_NODE = {
        "market_data": "data_loader",
        "fundamental_data": "data_loader",
        "tech_signals": "tech_analysis",
        "fundamental_signals": "fundamental_analysis",
        "sentiment_signals": "sentiment_analysis",
        "backtest_result": "backtest_runner",
        "feedback": "metrics_evaluator",
        "final_document": "document_writer",
        "status": "signal_fanout",  # signal_fanout 特有，且排在最后无冲突
    }

    def __init__(self, out_dir: Path, task_id: str = ""):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.task_id = task_id
        self.log_file = self.out_dir / "execution.md"
        self.status_file = self.out_dir / "status.json"
        # run_id -> {node_name, start_ts, event}
        self._run_map: Dict[str, Dict[str, Any]] = {}
        self.node_states: Dict[str, str] = {n: "pending" for n in self.NODE_ORDER}
        self.node_durations: Dict[str, int] = {}
        self.iteration: int = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_status("running", "DAG execution started")

    def _write_log(self, record: Dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def _write_status(self, status: str, message: str, extra: Dict | None = None) -> None:
        """实时写入 status.json，供外部进程查询进度"""
        data = {
            "task_id": self.task_id,
            "status": status,
            "message": message,
            "current_node": self._get_current_node(),
            "completed_nodes": sum(1 for s in self.node_states.values() if s == "success"),
            "total_nodes": len(self.NODE_ORDER),
            "node_states": self.node_states.copy(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.iteration > 0:
            data["iteration"] = self.iteration
        if extra:
            data.update(extra)
        self.status_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _get_current_node(self) -> Optional[str]:
        """返回当前正在运行的节点名"""
        for node in self.NODE_ORDER:
            if self.node_states.get(node) == "running":
                return node
        return None

    def _infer_node(self, outputs: Optional[Any]) -> Optional[str]:
        """根据 outputs 反推节点名。

        优先读取节点显式声明的 __node__ 字段（100% 准确），
        fallback 到类型/字段推断（兼容旧数据）。
        """
        # 1️⃣ 显式标记 — 节点自己声明身份（最可靠）
        if isinstance(outputs, dict) and "__node__" in outputs:
            return outputs["__node__"]

        # 2️⃣ 条件路由节点：返回字符串（如 "strategy_builder" / "document_writer"）
        if isinstance(outputs, str):
            return "refinement_router"

        if not isinstance(outputs, dict):
            return None

        # 3️⃣ LangGraph 最终状态：包含大量字段（>10 个），不是某个具体节点
        if len(outputs) > 10:
            return "graph_end"

        # 4️⃣ 条件路由节点：无输出字段
        if len(outputs) == 0:
            return "refinement_router"

        # 5️⃣ 字段推断（兼容旧数据/未标记的节点）
        for key, node in self._OUTPUT_TO_NODE.items():
            if key in outputs:
                return node

        # strategy_builder vs coding_node 歧义
        if "strategy_code" in outputs and "iteration" in outputs:
            return "strategy_builder"
        if "strategy_code" in outputs and "iteration" not in outputs:
            return "coding_node"

        return None

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id) if run_id else ""
        now = time.time()

        # 尝试从 serialized 获取节点名，失败则留空等 end 时推断
        node_name = None
        if serialized and "name" in serialized:
            node_name = serialized["name"]

        self._run_map[run_id_str] = {
            "node": node_name,
            "start_ts": now,
        }

        if node_name and node_name in self.NODE_ORDER:
            self.node_states[node_name] = "running"
            self._print_status()
            self._write_status("running", f"Node {node_name} started")

    def on_chain_end(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id) if run_id else ""
        now = time.time()
        run_info = self._run_map.pop(run_id_str, {})

        # 推断节点名
        node_name = run_info.get("node")
        if not node_name:
            node_name = self._infer_node(outputs)

        if not node_name:
            node_name = "unknown"

        start_ts = run_info.get("start_ts", now)
        duration_ms = int((now - start_ts) * 1000)

        if node_name in self.NODE_ORDER:
            self.node_states[node_name] = "success"
            self.node_durations[node_name] = duration_ms

        # 跟踪迭代次数
        if outputs and isinstance(outputs, dict) and "iteration" in outputs:
            self.iteration = outputs["iteration"]

        # ── 读取 LLM 摘要（1+2 组合方案） ──
        messages_digest = None
        messages_ref = None
        llm_log_dir = os.getenv("LLM_LOG_DIR")
        if llm_log_dir and node_name != "unknown":
            meta_path = Path(llm_log_dir) / f"{node_name}.meta.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    messages_digest = {
                        "messages_count": meta.get("messages_count", 2),
                        "prompt_chars": meta.get("prompt_chars", 0),
                        "response_chars": meta.get("response_chars", 0),
                        "prompt_preview": meta.get("prompt_preview", "")[:120],
                        "response_preview": meta.get("response_preview", "")[:120],
                    }
                    messages_ref = f"llm_logs/{node_name}.md"
                except Exception:
                    pass

        record = {
            "ts": now,
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "node": node_name,
            "event": "end",
            "duration_ms": duration_ms,
            "task_id": self.task_id,
            "output_keys": list(outputs.keys()) if isinstance(outputs, dict) else [],
        }
        if messages_digest:
            record["messages_digest"] = messages_digest
        if messages_ref:
            record["messages_ref"] = messages_ref
        self._write_log(record)

        if node_name in self.NODE_ORDER:
            self._print_status()
            # 更新 status.json：如果所有节点完成，标记 complete
            completed = sum(1 for s in self.node_states.values() if s == "success")
            if completed == len(self.NODE_ORDER):
                self._write_status("complete", "All nodes executed successfully", {
                    "completed_nodes": completed,
                    "total_nodes": len(self.NODE_ORDER),
                })
            else:
                self._write_status("running", f"Node {node_name} completed ({completed}/{len(self.NODE_ORDER)})")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id) if run_id else ""
        now = time.time()
        run_info = self._run_map.pop(run_id_str, {})
        node_name = run_info.get("node") or "unknown"
        start_ts = run_info.get("start_ts", now)
        duration_ms = int((now - start_ts) * 1000)

        if node_name in self.NODE_ORDER:
            self.node_states[node_name] = "error"
            self.node_durations[node_name] = duration_ms

        record = {
            "ts": now,
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "node": node_name,
            "event": "error",
            "error_type": type(error).__name__,
            "error_msg": str(error)[:500],
            "duration_ms": duration_ms,
            "task_id": self.task_id,
        }
        self._write_log(record)
        self._write_status("error", f"Node {node_name} failed: {str(error)[:200]}", {
            "error_node": node_name,
            "error_type": type(error).__name__,
        })
        self._print_status()

    def _print_status(self) -> None:
        """在 stderr 打印当前 DAG 状态（简洁版）"""
        lines = [f"\n[{self.task_id}] LangGraph 执行状态"]
        lines.append("=" * 50)
        for node in self.NODE_ORDER:
            label = self.NODE_LABELS.get(node, node)
            status = self.node_states.get(node, "pending")
            icon = {
                "pending": "⬜",
                "running": "🟡",
                "success": "✅",
                "error": "❌",
            }.get(status, "⬜")
            lines.append(f"{icon} {label}")
        lines.append("=" * 50)
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_states": self.node_states,
            "node_durations": self.node_durations,
            "completed": sum(1 for s in self.node_states.values() if s == "success"),
            "failed": sum(1 for s in self.node_states.values() if s == "error"),
            "total": len(self.NODE_ORDER),
        }
