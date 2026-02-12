from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@dataclass
class AgentResult:
    name: str
    output: str


class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def run(self, context: Dict[str, str]) -> AgentResult:
        raise NotImplementedError


class InterviewAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="采访代理",
            role="围绕人物经历进行结构化提问，提取关键人生信息。",
        )

    def run(self, context: Dict[str, str]) -> AgentResult:
        subject = context.get("subject_name", "受访者")
        era = context.get("era", "未填写")
        highlights = context.get("highlights", "未填写")
        raw_notes = context.get("raw_notes", "")

        questions = [
            f"{subject}的童年成长环境如何？有哪些影响一生的家庭记忆？",
            f"{subject}在{era}这个时代背景下，做过哪些关键选择？",
            f"{subject}最自豪和最艰难的经历分别是什么？",
            f"哪些价值观是{subject}最希望传给下一代的？",
        ]

        extracted = self._extract_bullets(raw_notes)
        summary = (
            "【采访提纲】\n- "
            + "\n- ".join(questions)
            + "\n\n【输入提炼】\n- "
            + "\n- ".join(extracted)
            + f"\n- 用户强调亮点：{highlights}"
        )
        return AgentResult(self.name, summary)

    @staticmethod
    def _extract_bullets(raw_notes: str) -> List[str]:
        if not raw_notes.strip():
            return ["暂无原始口述内容，建议先进行语音采访。"]
        cleaned = re.sub(r"\s+", " ", raw_notes).strip()
        parts = re.split(r"[。！？!?；;\n]", cleaned)
        bullets = [p.strip(" ，,.、") for p in parts if p.strip()]
        return bullets[:6] if bullets else [cleaned]


class StructuringAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="编排代理",
            role="将采访素材按时间线与主题线重组，生成章节框架。",
        )

    def run(self, context: Dict[str, str]) -> AgentResult:
        subject = context.get("subject_name", "受访者")
        interview_result = context.get("interview_result", "")
        tone = context.get("tone", "温暖真诚")

        chapters = [
            "第一章：家风与童年",
            "第二章：求学与立业",
            "第三章：家庭与责任",
            "第四章：转折与坚守",
            "第五章：晚年智慧与家训",
        ]

        outline = (
            f"为《{subject}人生口述史》生成章节骨架（文风：{tone}）：\n"
            + "\n".join(f"- {item}" for item in chapters)
            + "\n\n素材摘要：\n"
            + interview_result[:500]
        )
        return AgentResult(self.name, outline)


class WritingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="写作代理",
            role="基于采访提炼与章节框架，输出传记初稿。",
        )

    def run(self, context: Dict[str, str]) -> AgentResult:
        subject = context.get("subject_name", "受访者")
        tone = context.get("tone", "温暖真诚")
        generation_goal = context.get("generation_goal", "为家庭留下一份可传承的人生记录")
        interview_result = context.get("interview_result", "")

        draft = f"""《{subject}传》\n\n序言\n这是一份以{tone}笔触写成的生命记录，目标是{generation_goal}。\n\n正文（节选）\n{subject}的一生，是普通人奋斗与守望的缩影。从家庭记忆到时代转折，从责任承担到价值传承，每一步都印证了“平凡亦可伟大”。\n\n根据采访信息可见：\n{interview_result[:350]}\n\n结语\n愿这份传记成为家族的精神灯塔，让后代在阅读中看见来路、获得力量。\n"""
        return AgentResult(self.name, draft)


class BiographyMultiAgentPipeline:
    def __init__(self) -> None:
        self.interview_agent = InterviewAgent()
        self.structuring_agent = StructuringAgent()
        self.writing_agent = WritingAgent()

    def run(self, payload: Dict[str, str]) -> Dict[str, str]:
        interview = self.interview_agent.run(payload)
        payload["interview_result"] = interview.output

        structure = self.structuring_agent.run(payload)
        payload["structure_result"] = structure.output

        writing = self.writing_agent.run(payload)

        return {
            "interview": interview.output,
            "structure": structure.output,
            "biography": writing.output,
        }


pipeline = BiographyMultiAgentPipeline()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    payload = {
        "subject_name": request.form.get("subject_name", "").strip(),
        "era": request.form.get("era", "").strip(),
        "highlights": request.form.get("highlights", "").strip(),
        "raw_notes": request.form.get("raw_notes", "").strip(),
        "tone": request.form.get("tone", "").strip() or "温暖真诚",
        "generation_goal": request.form.get("generation_goal", "").strip(),
    }

    if not payload["subject_name"]:
        return render_template("index.html", error="请至少填写人物姓名。", payload=payload)

    results = pipeline.run(payload)

    if request.headers.get("Accept") == "application/json":
        return jsonify(results)

    return render_template("index.html", results=results, payload=payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
