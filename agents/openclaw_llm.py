"""
OpenClaw LLM 桥接层

通过 OpenClaw CLI 调用 Agent，复用已有的 provider 配置和智能体定义。
所有角色默认映射到 'developer' Agent，可通过 OPENCLAW_ROLE_MAP 环境变量自定义。
"""

import json
import os
import subprocess
import sys
import tempfile
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

    # 角色 -> OpenClaw Agent ID 的映射（默认全部用 player）
    ROLE_AGENT_MAP = {
        "tech_analyst": "player",
        "fundamental_analyst": "player",
        "sentiment_analyst": "player",
        "strategy_dev": "player",
        "document_writer": "player",
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
        agent_id = self.ROLE_AGENT_MAP.get(self.agent_role, "player")
        turn_timeout = int(os.getenv("OPENCLAW_TIMEOUT", "120"))

        # 根据角色调整 prompt 后缀
        if self.agent_role == "strategy_dev":
            # 策略开发节点：要求输出 JSON 配置
            constrained_prompt = prompt + "\n\n【重要】只输出合法的 JSON 对象，不要任何解释文字。"
        elif self.agent_role == "document_writer":
            # 文档节点：正常输出
            constrained_prompt = prompt
        else:
            # 分析节点：简洁输出
            constrained_prompt = prompt + "\n\n【重要】请简洁回答，不要表格和 markdown，不超过 500 字。"

        # stdout 重定向到临时文件
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'w', encoding='utf-8') as out:
                result = subprocess.run(
                    [
                        "openclaw", "agent",
                        "--agent", agent_id,
                        "--message", constrained_prompt,
                        "--timeout", str(turn_timeout),
                    ],
                    stdout=out,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=turn_timeout + 20,
                )

            if result.returncode != 0:
                raise RuntimeError(
                    f"openclaw agent --agent {agent_id} failed "
                    f"(exit {result.returncode}): {result.stderr[:400]}"
                )

            with open(tmp_path, 'r', encoding='utf-8') as f:
                raw = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 过滤掉插件加载日志，只保留实际响应内容
        lines = raw.splitlines()
        # 找到最后一个非日志行（以 [plugins] 开头的是日志）
        content_lines = [l for l in lines if not l.startswith('[plugins]')]
        text = '\n'.join(content_lines).strip()

        if not text:
            raise RuntimeError(f"No content in openclaw response for agent {agent_id}")

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
