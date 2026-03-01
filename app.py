from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from agents import BiographyMultiAgentFramework
from agents.framework import AgendaStore, SessionAgendaItem

app = Flask(__name__)


@dataclass
class ChatSession:
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    ready_to_write: bool = False
    biography: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    agenda: AgendaStore = field(default_factory=AgendaStore)
    asked_item: SessionAgendaItem | None = None
    user_topics: List[str] = field(default_factory=list)


framework: BiographyMultiAgentFramework | None = None
sessions: Dict[str, ChatSession] = {}


def get_framework() -> BiographyMultiAgentFramework:
    global framework
    if framework is None:
        framework = BiographyMultiAgentFramework(cfg_path="config/agents.yaml")
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
    greeting = "你好，我是 StorySage 采访助手。先告诉我：这次你最想聊哪一段人生经历？"
    session = ChatSession(
        session_id=session_id,
        messages=[{"role": "assistant", "content": greeting}],
    )
    session.agenda = get_framework().prepare_session(user_topics=[], previous_summary=[])
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
    turn_id = str(uuid.uuid4())

    scribe_result = get_framework().process_user_turn(
        session_id=session_id,
        turn_id=turn_id,
        asked_item=session.asked_item,
        user_answer=user_message,
        agenda=session.agenda,
    )

    if wants_to_finish(user_message):
        pipeline_result = get_framework().run_biography_update(session.messages)
        session.ready_to_write = True
        session.biography = pipeline_result
        session.logs = pipeline_result["plans"]
        session.messages.append(
            {
                "role": "assistant",
                "content": "收到，我们结束采访并完成本轮 Biography 更新。",
            }
        )
    else:
        interview_result = get_framework().interview_turn(
            session_id=session_id,
            history=session.messages,
            agenda=session.agenda,
        )
        session.asked_item = interview_result["asked_item"]
        session.messages.append({"role": "assistant", "content": interview_result["assistant"]})

        if len(get_framework().memory_bank.get_unprocessed()) >= 10:
            pipeline_result = get_framework().run_biography_update(session.messages)
            session.logs.extend(pipeline_result["plans"])
            session.biography = pipeline_result

    return jsonify(
        {
            "session_id": session.session_id,
            "messages": session.messages,
            "ready_to_write": session.ready_to_write,
            "logs": session.logs,
            "result": session.biography,
            "scribe": {
                "new_memories": len(scribe_result["new_memories"]),
                "followups": scribe_result["followups"],
            },
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

    pipeline_result = get_framework().run_biography_update(session.messages)
    session.ready_to_write = True
    session.biography = pipeline_result
    session.logs = pipeline_result["plans"]
    session.messages.append(
        {
            "role": "assistant",
            "content": "采访已结束，系统已根据未处理记忆完成写作更新。",
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
