from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from openai import OpenAI


@dataclass
class AgentResult:
    name: str
    output: str


@dataclass
class MemoryEntity:
    memory_id: str
    text: str
    date: str | None = None
    location: str | None = None
    people: List[str] = field(default_factory=list)
    emotion: str | None = None
    source_turn_id: str = ""
    source_session_id: str = ""
    embedding: List[float] = field(default_factory=list)
    processed: bool = False


@dataclass
class QuestionEntry:
    question_id: str
    question_text: str
    answer_turn_id: str
    session_id: str
    embedding: List[float]
    timestamp: int


@dataclass
class SessionAgendaItem:
    agenda_id: str
    question_text: str
    status: str = "proposed"
    source: str = "coordinator"
    topic_tag: str = "general"
    response_summary: str = ""


@dataclass
class SectionNode:
    path: str
    title: str
    content: str
    children: List["SectionNode"] = field(default_factory=list)


@dataclass
class BiographyDocument:
    sections: Dict[str, SectionNode] = field(default_factory=dict)
    edit_comments: List[str] = field(default_factory=list)


@dataclass
class UpdatePlan:
    plan_id: str
    action: str
    section_path: str
    title: str
    guidance: str
    referenced_memories: List[str]


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

    def chat(self, model: str, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
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


class MemoryBank:
    def __init__(self):
        self.memories: Dict[str, MemoryEntity] = {}

    def add(self, memory: MemoryEntity) -> None:
        self.memories[memory.memory_id] = memory

    def mark_processed(self, memory_ids: Sequence[str]) -> None:
        for memory_id in memory_ids:
            if memory_id in self.memories:
                self.memories[memory_id].processed = True

    def get_unprocessed(self) -> List[MemoryEntity]:
        return [m for m in self.memories.values() if not m.processed]

    def recall(self, query: str, top_k: int = 5) -> List[MemoryEntity]:
        query_vec = embed_text(query)
        scored: List[Tuple[float, MemoryEntity]] = []
        for memory in self.memories.values():
            scored.append((cosine_similarity(query_vec, memory.embedding), memory))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k] if item[0] > 0.15]


class QuestionBank:
    def __init__(self):
        self.questions: Dict[str, QuestionEntry] = {}
        self._counter = 0

    def add(self, question_text: str, answer_turn_id: str, session_id: str) -> QuestionEntry:
        self._counter += 1
        entry = QuestionEntry(
            question_id=f"Q{self._counter}",
            question_text=question_text,
            answer_turn_id=answer_turn_id,
            session_id=session_id,
            embedding=embed_text(question_text),
            timestamp=self._counter,
        )
        self.questions[entry.question_id] = entry
        return entry

    def is_duplicate(self, question_text: str, threshold: float = 0.8, top_k: int = 3) -> bool:
        q_vec = embed_text(question_text)
        scored: List[float] = []
        for q in self.questions.values():
            scored.append(cosine_similarity(q_vec, q.embedding))
        scored.sort(reverse=True)
        return any(score >= threshold for score in scored[:top_k])


class AgendaStore:
    def __init__(self):
        self.items: List[SessionAgendaItem] = []

    def add_question(self, question: str, source: str, topic_tag: str = "general") -> SessionAgendaItem:
        item = SessionAgendaItem(agenda_id=str(uuid.uuid4()), question_text=question, source=source, topic_tag=topic_tag)
        self.items.append(item)
        return item

    def next_item(self) -> Optional[SessionAgendaItem]:
        for item in self.items:
            if item.status == "proposed":
                return item
        return None

    def mark_answered(self, agenda_id: str, response_summary: str) -> None:
        for item in self.items:
            if item.agenda_id == agenda_id:
                item.status = "answered"
                item.response_summary = response_summary
                return


class InterviewerAgent(BaseAgent):
    def next_question(
        self,
        dialogue_state: List[Dict[str, str]],
        session_agenda: AgendaStore,
        recalled_memories: List[MemoryEntity],
    ) -> str:
        item = session_agenda.next_item()
        if item:
            item.status = "asked"
            memory_hints = "\n".join([f"- {m.text}" for m in recalled_memories[:3]]) or "- 暂无相关旧记忆"
            result = self.run(
                dialogue_state
                + [
                    {
                        "role": "system",
                        "content": f"优先围绕这个议程问题：{item.question_text}\n可参考旧记忆:\n{memory_hints}",
                    }
                ]
            )
            return result.output

        result = self.run(dialogue_state + [{"role": "system", "content": "没有现成议程时，自然生成一个开放追问。"}])
        return result.output


class SessionScribeAgent(BaseAgent):
    def process_turn(
        self,
        session_id: str,
        turn_id: str,
        asked_question: SessionAgendaItem | None,
        user_answer: str,
        memory_bank: MemoryBank,
        question_bank: QuestionBank,
        agenda: AgendaStore,
    ) -> Dict[str, Any]:
        memories = self._decompose_memory(session_id=session_id, turn_id=turn_id, user_answer=user_answer)
        for m in memories:
            memory_bank.add(m)

        implicit_qs = self._extract_implicit_questions(user_answer)
        for q in implicit_qs:
            question_bank.add(question_text=q, answer_turn_id=turn_id, session_id=session_id)

        if asked_question:
            summary = summarize_text(user_answer)
            agenda.mark_answered(asked_question.agenda_id, response_summary=summary)

        followups = self._propose_followups(user_answer, memories)
        accepted_followups: List[str] = []
        for fup in followups:
            if not question_bank.is_duplicate(fup, top_k=3):
                agenda.add_question(fup, source="scribe")
                accepted_followups.append(fup)

        return {
            "new_memories": memories,
            "implicit_qs": implicit_qs,
            "followups": accepted_followups,
            "agenda_updates": [item.__dict__ for item in agenda.items],
        }

    def _decompose_memory(self, session_id: str, turn_id: str, user_answer: str) -> List[MemoryEntity]:
        chunks = [s.strip() for s in re.split(r"[。！？!?]\s*", user_answer) if s.strip()]
        memories: List[MemoryEntity] = []
        for chunk in chunks[:3]:
            memories.append(
                MemoryEntity(
                    memory_id=str(uuid.uuid4()),
                    text=chunk,
                    source_turn_id=turn_id,
                    source_session_id=session_id,
                    embedding=embed_text(chunk),
                    people=extract_people(chunk),
                    emotion=extract_emotion(chunk),
                )
            )
        return memories

    def _extract_implicit_questions(self, user_answer: str) -> List[str]:
        base_questions = [
            "当时发生这件事的背景是什么？",
            "这件事对你后来的选择产生了什么影响？",
            "有没有一个让你印象最深的细节？",
        ]
        if len(user_answer) < 40:
            return base_questions[:1]
        return base_questions

    def _propose_followups(self, user_answer: str, memories: Sequence[MemoryEntity]) -> List[str]:
        factual = "你提到的关键事件发生在什么时间和地点？"
        reflective = "回头看，这段经历最改变你的是什么？"
        relation = "这段经历和你后来的人生阶段有什么联系？"
        if len(memories) <= 1:
            return [factual]
        if "家" in user_answer or "父" in user_answer or "母" in user_answer:
            return [factual, reflective, relation, "家人当时的反应如何？"]
        return [factual, reflective, relation]


class PlannerAgent(BaseAgent):
    def make_plans(
        self,
        biography: BiographyDocument,
        new_memories: Sequence[MemoryEntity],
        user_edits: Sequence[str],
    ) -> Tuple[List[UpdatePlan], List[MemoryEntity]]:
        groups = self._group_memories(new_memories)
        plans: List[UpdatePlan] = []
        covered: set[str] = set()

        for idx, group in enumerate(groups, start=1):
            memory_ids = [m.memory_id for m in group]
            covered.update(memory_ids)
            section_path = f"{idx}"
            title = infer_title(group)
            guidance = f"围绕主题“{title}”按时间推进，保留关键细节和人物互动。"
            action = "ReviseSection" if section_path in biography.sections else "CreateNewSection"
            plans.append(
                UpdatePlan(
                    plan_id=str(uuid.uuid4()),
                    action=action,
                    section_path=section_path,
                    title=title,
                    guidance=guidance,
                    referenced_memories=memory_ids,
                )
            )

        for comment in user_edits:
            plans.append(
                UpdatePlan(
                    plan_id=str(uuid.uuid4()),
                    action="ReviseSection",
                    section_path="editor-note",
                    title="编辑意见修订",
                    guidance=f"吸收用户编辑意见并改写相关段落：{comment}",
                    referenced_memories=[],
                )
            )

        uncovered = [m for m in new_memories if m.memory_id not in covered]
        if uncovered:
            rescue_ids = [m.memory_id for m in uncovered]
            plans.append(
                UpdatePlan(
                    plan_id=str(uuid.uuid4()),
                    action="CreateNewSection",
                    section_path=str(len(plans) + 1),
                    title="补充回忆",
                    guidance="将遗漏记忆整合为补充章节，避免丢失。",
                    referenced_memories=rescue_ids,
                )
            )
            uncovered = []

        return plans, uncovered

    def _group_memories(self, memories: Sequence[MemoryEntity]) -> List[List[MemoryEntity]]:
        grouped: Dict[str, List[MemoryEntity]] = {}
        for m in memories:
            key = m.people[0] if m.people else "个人成长"
            grouped.setdefault(key, []).append(m)
        return list(grouped.values())


class SectionWriterAgent(BaseAgent):
    def execute_plan(
        self,
        plan: UpdatePlan,
        memory_bank: MemoryBank,
        dialogue_snippets: Sequence[Dict[str, str]],
    ) -> str:
        referenced = [memory_bank.memories[m_id] for m_id in plan.referenced_memories if m_id in memory_bank.memories]
        context = "\n".join([f"- {m.text}" for m in referenced]) or "- 按编辑意见改写"
        latest_dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in dialogue_snippets[-4:]])

        result = self.run(
            [
                {
                    "role": "user",
                    "content": (
                        f"章节路径：{plan.section_path}\n"
                        f"标题：{plan.title}\n"
                        f"写作指导：{plan.guidance}\n"
                        f"相关记忆：\n{context}\n"
                        f"最近对话片段：\n{latest_dialogue}"
                    ),
                }
            ]
        )
        return result.output


class SessionCoordinatorAgent(BaseAgent):
    def build_agenda(
        self,
        prev_summaries: Sequence[str],
        user_topics: Sequence[str],
        followups: Sequence[str],
    ) -> AgendaStore:
        agenda = AgendaStore()
        for topic in user_topics:
            agenda.add_question(f"你希望继续聊“{topic}”的哪段经历？", source="coordinator", topic_tag=topic)
        for question in followups:
            agenda.add_question(question, source="coordinator")
        if not agenda.items:
            agenda.add_question("我们从你人生中一个重要转折点开始聊起，好吗？", source="coordinator")
        return agenda


class BiographyMultiAgentFramework:
    def __init__(self, cfg_path: str = "config/agents.yaml"):
        cfg = OpenAIConfig(cfg_path)
        llm = OpenAILLM(cfg)
        prompts = cfg.prompts
        models = cfg.models

        self.interviewer = InterviewerAgent(
            name="Interviewer",
            llm=llm,
            model=models.get("interviewer", "gpt-4o-mini"),
            prompt=prompts.get("interviewer", "你是采访代理。"),
        )
        self.scribe = SessionScribeAgent(
            name="SessionScribe",
            llm=llm,
            model=models.get("scribe", "gpt-4o-mini"),
            prompt=prompts.get("scribe", "你是会话记录代理。"),
        )
        self.planner = PlannerAgent(
            name="Planner",
            llm=llm,
            model=models.get("planner", "gpt-4o-mini"),
            prompt=prompts.get("planner", "你是写作规划代理。"),
        )
        self.section_writer = SectionWriterAgent(
            name="SectionWriter",
            llm=llm,
            model=models.get("section_writer", "gpt-4o"),
            prompt=prompts.get("section_writer", "你负责章节写作。"),
        )
        self.coordinator = SessionCoordinatorAgent(
            name="SessionCoordinator",
            llm=llm,
            model=models.get("coordinator", "gpt-4o-mini"),
            prompt=prompts.get("coordinator", "你是会话协调代理。"),
        )

        self.memory_bank = MemoryBank()
        self.question_bank = QuestionBank()
        self.biography = BiographyDocument()
        self.pending_followups: List[str] = []

    def prepare_session(self, user_topics: Sequence[str], previous_summary: Sequence[str]) -> AgendaStore:
        return self.coordinator.build_agenda(previous_summary, user_topics, self.pending_followups)

    def interview_turn(self, session_id: str, history: List[Dict[str, str]], agenda: AgendaStore) -> Dict[str, Any]:
        recalled = self.memory_bank.recall(history[-1]["content"] if history else "")
        assistant_question = self.interviewer.next_question(history, agenda, recalled)
        asked_item = next((item for item in agenda.items if item.status == "asked"), None)
        return {
            "assistant": assistant_question,
            "asked_item": asked_item,
            "recalled": [m.text for m in recalled],
        }

    def process_user_turn(
        self,
        session_id: str,
        turn_id: str,
        asked_item: SessionAgendaItem | None,
        user_answer: str,
        agenda: AgendaStore,
    ) -> Dict[str, Any]:
        scribe_result = self.scribe.process_turn(
            session_id=session_id,
            turn_id=turn_id,
            asked_question=asked_item,
            user_answer=user_answer,
            memory_bank=self.memory_bank,
            question_bank=self.question_bank,
            agenda=agenda,
        )
        self.pending_followups.extend(scribe_result["followups"])
        return scribe_result

    def run_biography_update(self, history: List[Dict[str, str]], user_edits: Sequence[str] | None = None) -> Dict[str, Any]:
        user_edits = user_edits or []
        unprocessed = self.memory_bank.get_unprocessed()
        plans, uncovered = self.planner.make_plans(self.biography, unprocessed, user_edits)

        plan_logs = []
        for plan in plans:
            content = self.section_writer.execute_plan(plan, self.memory_bank, history)
            self.biography.sections[plan.section_path] = SectionNode(path=plan.section_path, title=plan.title, content=content)
            self.memory_bank.mark_processed(plan.referenced_memories)
            plan_logs.append(
                {
                    "plan_id": plan.plan_id,
                    "action": plan.action,
                    "section_path": plan.section_path,
                    "title": plan.title,
                    "memory_count": len(plan.referenced_memories),
                }
            )

        return {
            "plans": plan_logs,
            "uncovered_memories": [m.memory_id for m in uncovered],
            "sections": [
                {"path": node.path, "title": node.title, "content": node.content}
                for node in self.biography.sections.values()
            ],
        }


def embed_text(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    if not text:
        return vec
    for token in tokenize(text):
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def tokenize(text: str) -> Iterable[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def summarize_text(text: str, max_len: int = 60) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")


def extract_people(text: str) -> List[str]:
    tags = []
    mapping = {
        "父亲": ["父", "爸爸", "父亲"],
        "母亲": ["母", "妈妈", "母亲"],
        "老师": ["老师", "导师"],
        "朋友": ["朋友", "同学"],
    }
    for label, aliases in mapping.items():
        if any(alias in text for alias in aliases):
            tags.append(label)
    return tags


def extract_emotion(text: str) -> str | None:
    if any(w in text for w in ["开心", "高兴", "幸福"]):
        return "positive"
    if any(w in text for w in ["难过", "痛苦", "焦虑", "害怕"]):
        return "negative"
    return None


def infer_title(memories: Sequence[MemoryEntity]) -> str:
    if not memories:
        return "未命名章节"
    first = memories[0].text
    return summarize_text(first, max_len=16)
