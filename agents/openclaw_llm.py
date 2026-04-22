"""
OpenClaw LLM 桥接层

通过 OpenClaw CLI 调用 Agent，复用已有的 provider 配置和智能体定义。
所有角色默认映射到 'developer' Agent，可通过 OPENCLAW_ROLE_MAP 环境变量自定义。
"""

import json
import os
import subprocess
import sys
from typing import Optional


class LLMInterface:
    """LLM 调用抽象层"""

    def __init__(self, agent_role: str = "default"):
        self.agent_role = agent_role

    def call(self, prompt: str) -> str:
        raise NotImplementedError


class OpenClawLLM(LLMInterface):
    """
    通过 OpenClaw CLI 调用 Agent。
    复用 OpenClaw 已有的 provider 配置和智能体定义，无需额外 API Key。
    """

    # 角色 -> OpenClaw Agent ID 的映射（默认全部用 developer）
    ROLE_AGENT_MAP = {
        "tech_analyst": "developer",
        "fundamental_analyst": "developer",
        "sentiment_analyst": "developer",
        "strategy_dev": "developer",
        "document_writer": "developer",
    }

    def __init__(self, agent_role: str = "default"):
        super().__init__(agent_role)
        # 允许通过环境变量自定义映射
        env_map = os.getenv("OPENCLAW_ROLE_MAP", "")
        if env_map:
            for pair in env_map.split(","):
                if "=" in pair:
                    role, agent_id = pair.split("=", 1)
                    self.ROLE_AGENT_MAP[role.strip()] = agent_id.strip()

    def call(self, prompt: str) -> str:
        agent_id = self.ROLE_AGENT_MAP.get(self.agent_role, "developer")
        turn_timeout = int(os.getenv("OPENCLAW_TIMEOUT", "120"))

        result = subprocess.run(
            [
                "openclaw", "agent",
                "--agent", agent_id,
                "--message", prompt,
                "--json",
                "--timeout", str(turn_timeout),
            ],
            capture_output=True,
            text=True,
            timeout=turn_timeout + 20,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"openclaw agent --agent {agent_id} failed "
                f"(exit {result.returncode}): {result.stderr[:400]}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON from openclaw agent {agent_id}: {exc}\n"
                f"Raw output: {result.stdout[:500]}"
            ) from exc

        payloads = data.get("result", {}).get("payloads", [])
        if not payloads:
            raise RuntimeError(f"No payloads in openclaw response for agent {agent_id}")

        text = payloads[0].get("text", "")
        aborted = data.get("result", {}).get("meta", {}).get("aborted", False)
        if aborted:
            raise RuntimeError(f"Agent {agent_id} turn was aborted by runtime")
        return text


def get_llm(agent_role: str) -> LLMInterface:
    """
    获取指定角色的 LLM 实例。
    通过 OpenClaw CLI 调用 Agent，复用已有 provider 配置。
    """
    try:
        return OpenClawLLM(agent_role=agent_role)
    except Exception as e:
        raise RuntimeError(
            f"OpenClaw LLM 初始化失败: {e}\n"
            f"请确认 openclaw 命令已安装且可用。"
        ) from e
