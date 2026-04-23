"""
LLM 调用日志工具

提供 call_llm_with_log() 包装函数，在调用 LLM 的同时：
1. 把完整 prompt + response 写入 llm_logs/{role}.md（给人看的 markdown）
2. 生成摘要 meta.json（给 tracing.py 读取嵌入 execution.md）

用法：
    from utils.llm_logger import call_llm_with_log
    response = call_llm_with_log("tech_analyst", prompt)

环境变量：
    LLM_LOG_DIR — 日志目录路径（由 strategy_mas.py 的 main() 设置）
"""

import json
import os
from pathlib import Path
from typing import Optional

from agents.openclaw_llm import get_llm


def call_llm_with_log(role: str, prompt: str) -> str:
    """
    调用 LLM 并记录完整对话日志。

    如果 LLM_LOG_DIR 环境变量已设置，则：
    - 写完整对话到 {LLM_LOG_DIR}/{role}.md
    - 写摘要到 {LLM_LOG_DIR}/{role}.meta.json
    """
    llm = get_llm(role)
    response = llm.call(prompt)

    log_dir = os.getenv("LLM_LOG_DIR")
    if not log_dir:
        return response

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 写完整日志（markdown，给人看，追加模式） ──
    md_path = log_dir / f"{role}.md"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"\n---\n\n# {role} LLM Call\n\n")
        f.write(f"## Prompt ({len(prompt)} chars)\n\n")
        f.write("```\n")
        f.write(prompt)
        f.write("\n```\n\n")
        f.write(f"## Response ({len(response)} chars)\n\n")
        f.write("```\n")
        f.write(response)
        f.write("\n```\n")

    # ── 写摘要（JSON，给 tracing.py 嵌入 execution.md） ──
    meta = {
        "role": role,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "prompt_preview": prompt[:300],
        "response_preview": response[:300],
        "messages_count": 2,  # prompt + response 一轮对话
    }
    meta_path = log_dir / f"{role}.meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return response
