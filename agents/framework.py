from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI


@dataclass
class AgentResult:
    name: str
    output: str


class OpenAIConfig:
    def __init__(self, config_path: str = "config/agents.yaml"):
        self.path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"配置文件不存在：{self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY") or self.config.get("openai", {}).get("api_key", "")
        return key.strip()

    @property
    def base_url(self) -> str:
        return (self.config.get("openai", {}).get("base_url", "https://api.openai.com/v1") or "").strip()

    @property
    def models(self) -> Dict[str, str]:
        return self.config.get("models", {})

    @property
    def prompts(self) -> Dict[str, str]:
        return self.config.get("prompts", {})


class OpenAILLM:
    def __init__(self, cfg: OpenAIConfig):
        if not cfg.api_key:
            raise ValueError("缺少 OPENAI_API_KEY，请在环境变量或 config/agents.yaml 中填写。")
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, model: str, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        resp = self.client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()


class BaseAgent:
    def __init__(self, name: str, llm: OpenAILLM, model: str, prompt: str):
        self.name = name
        self.llm = llm
        self.model = model
        self.prompt = prompt

    def run(self, messages: List[Dict[str, str]]) -> AgentResult:
        output = self.llm.chat(model=self.model, system_prompt=self.prompt, messages=messages)
        return AgentResult(self.name, output)


class InterviewAgent(BaseAgent):
    pass


class SufficiencyAgent(BaseAgent):
    def run(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        result = super().run(messages)
        ready = result.output.startswith("READY:")
        return ready, result.output


class StructuringAgent(BaseAgent):
    pass


class WritingAgent(BaseAgent):
    pass


class BiographyMultiAgentFramework:
    def __init__(self, cfg_path: str = "config/agents.yaml"):
        cfg = OpenAIConfig(cfg_path)
        llm = OpenAILLM(cfg)
        prompts = cfg.prompts
        models = cfg.models

        self.interview_agent = InterviewAgent(
            name="采访代理",
            llm=llm,
            model=models.get("interview", "gpt-4o-mini"),
            prompt=prompts.get("interview", "你是采访代理。"),
        )
        self.sufficiency_agent = SufficiencyAgent(
            name="充足性判断代理",
            llm=llm,
            model=models.get("sufficiency", "gpt-4o-mini"),
            prompt=prompts.get("sufficiency", "你负责判断素材是否充足。"),
        )
        self.structuring_agent = StructuringAgent(
            name="编排代理",
            llm=llm,
            model=models.get("structure", "gpt-4o-mini"),
            prompt=prompts.get("structure", "你负责生成章节结构。"),
        )
        self.writing_agent = WritingAgent(
            name="写作代理",
            llm=llm,
            model=models.get("writing", "gpt-4o-mini"),
            prompt=prompts.get("writing", "你负责写作传记初稿。"),
        )

    def interview_turn(self, history: List[Dict[str, str]]) -> str:
        interview = self.interview_agent.run(history)
        return interview.output

    def generate_biography(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        logs: List[Dict[str, str]] = []

        ready, judge_text = self.sufficiency_agent.run(history)
        logs.append(
            {
                "agent": self.sufficiency_agent.name,
                "status": "done",
                "output": judge_text,
            }
        )

        if not ready:
            logs.append(
                {
                    "agent": "流程控制",
                    "status": "warning",
                    "output": "素材尚未完全充足，但已根据你的要求进入写作流程。",
                }
            )

        structure = self.structuring_agent.run(history)
        logs.append(
            {
                "agent": self.structuring_agent.name,
                "status": "done",
                "output": structure.output,
            }
        )

        writing = self.writing_agent.run(history + [{"role": "assistant", "content": structure.output}])
        logs.append(
            {
                "agent": self.writing_agent.name,
                "status": "done",
                "output": "传记初稿生成完成。",
            }
        )

        return {
            "logs": logs,
            "result": {
                "structure": structure.output,
                "biography": writing.output,
            },
        }
