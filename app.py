from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

from agents import BiographyMultiAgentFramework

app = Flask(__name__)


@dataclass
class ChatSession:
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    ready_to_write: bool = False
    biography: Dict[str, str] | None = None
    logs: List[Dict[str, str]] = field(default_factory=list)


framework: BiographyMultiAgentFramework | None = None
sessions: Dict[str, ChatSession] = {}


def get_framework() -> BiographyMultiAgentFramework:
    global framework
    if framework is None:
        framework = BiographyMultiAgentFramework(cfg_path="/Users/duanfeiyu/Documents/Personal_Bio/config.yaml")
    return framework


def wants_to_finish(message: str) -> bool:
    text = message.strip().lower()
    keywords = ["开始写", "生成传记", "结束采访", "可以写了", "开始生成", "finish", "write"]
    return any(k in text for k in keywords)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/chat/start", methods=["POST"])
def chat_start():
    session_id = str(uuid.uuid4())
    greeting = "你好，我是你的传记采访助手。我们先从你的讲述对象开始：TA是谁，你和TA是什么关系？"
    session = ChatSession(
        session_id=session_id,
        messages=[{"role": "assistant", "content": greeting}],
    )
    sessions[session_id] = session
    return jsonify({"session_id": session_id, "messages": session.messages})


@app.route("/chat/message", methods=["POST"])
def chat_message():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    user_message = (data.get("message") or "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "会话不存在，请刷新后重新开始。"}), 400
    if not user_message:
        return jsonify({"error": "请输入消息内容。"}), 400

    session = sessions[session_id]
    if session.ready_to_write:
        return jsonify({"error": "该会话已完成写作，如需继续请开启新会话。"}), 400

    session.messages.append({"role": "user", "content": user_message})

    if wants_to_finish(user_message):
        try:
            pipeline_result = get_framework().generate_biography(session.messages)
        except Exception as e:
            return jsonify({"error": f"LLM 初始化或调用失败：{e}"}), 500

        session.ready_to_write = True
        session.biography = pipeline_result["result"]
        session.logs = pipeline_result["logs"]
        session.messages.append(
            {
                "role": "assistant",
                "content": "收到，我们现在结束采访，开始多代理协作生成传记。",
            }
        )
    else:
        try:
            assistant_text = get_framework().interview_turn(session.messages)
        except Exception as e:
            return jsonify({"error": f"LLM 初始化或调用失败：{e}"}), 500
        session.messages.append({"role": "assistant", "content": assistant_text})

    return jsonify(
        {
            "session_id": session.session_id,
            "messages": session.messages,
            "ready_to_write": session.ready_to_write,
            "logs": session.logs,
            "result": session.biography,
        }
    )


@app.route("/chat/finish", methods=["POST"])
def chat_finish():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "会话不存在，请刷新后重新开始。"}), 400

    session = sessions[session_id]
    if session.ready_to_write:
        return jsonify(
            {
                "session_id": session.session_id,
                "messages": session.messages,
                "ready_to_write": session.ready_to_write,
                "logs": session.logs,
                "result": session.biography,
            }
        )

    try:
        pipeline_result = get_framework().generate_biography(session.messages)
    except Exception as e:
        return jsonify({"error": f"LLM 初始化或调用失败：{e}"}), 500

    session.ready_to_write = True
    session.biography = pipeline_result["result"]
    session.logs = pipeline_result["logs"]
    session.messages.append(
        {
            "role": "assistant",
            "content": "采访已结束，正在侧边展示多代理生成结果。",
        }
    )

    return jsonify(
        {
            "session_id": session.session_id,
            "messages": session.messages,
            "ready_to_write": session.ready_to_write,
            "logs": session.logs,
            "result": session.biography,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
