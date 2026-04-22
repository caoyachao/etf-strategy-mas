"""
输出管理器

管理 MAS 工作流的文件输出：日志、状态、策略代码、报告。
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


class OutputManager:
    def __init__(self, output_dir: Path, task_id: str):
        self.output_dir = output_dir
        self.task_id = task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._transcript_path = output_dir / "transcript.md"
        self._status_path = output_dir / "status.json"
        self._result_path = output_dir / "result.md"
        self._strategy_path = output_dir / "strategy.py"
        self._report_path = output_dir / "report.md"

        self._transcript_path.write_text(f"# Transcript — {task_id}\n\n")
        self._write_status("running", "")

    def log(self, node_name: str, content: str, node_type: str = "node") -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"## [{ts}] {node_name.upper()} ({node_type})\n\n{content}\n\n---\n\n"
        with open(self._transcript_path, "a") as fh:
            fh.write(entry)
        preview = content[:80].replace("\n", " ")
        print(f"[{ts}] [{node_name}] {preview}...", file=sys.stderr, flush=True)

    def _write_status(self, status: str, message: str, extra: Dict | None = None) -> None:
        data = {
            "task_id": self.task_id,
            "status": status,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            data.update(extra)
        self._status_path.write_text(json.dumps(data, indent=2))

    def save_strategy(self, code: str) -> None:
        self._strategy_path.write_text(code)

    def save_report(self, report: str) -> None:
        self._report_path.write_text(report)

    def complete(self, result: str, steps: int, metrics: Dict) -> None:
        self._write_status("complete", f"Completed in {steps} iterations", {"steps": steps, "metrics": metrics})
        fm = (
            f"---\n"
            f"task_id: {self.task_id}\n"
            f"status: complete\n"
            f"iterations: {steps}\n"
            f"timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )
        self._result_path.write_text(fm + result)

    def error(self, exc: Exception, steps: int, tb_str: str = "") -> None:
        msg = str(exc)[:400]
        self._write_status("error", msg, {"steps": steps})
        fm = (
            f"---\n"
            f"task_id: {self.task_id}\n"
            f"status: error\n"
            f"steps: {steps}\n"
            f"timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )
        self._result_path.write_text(fm + f"ERROR: {msg}\n\n```\n{tb_str or msg}\n```\n")
