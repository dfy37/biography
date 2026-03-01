# StorySage Web Demo（Flask + Five-Agent Workflow）

这是一个基于 Flask 的 StorySage 复现 Demo：把“采访-写作-下轮准备”拆成三阶段，并通过共享数据结构协作。

## 复现要点
- 三阶段：
  - Interview Session（Interviewer + Session Scribe）
  - Biography Writing（Planner + Section Writer）
  - Subsequent Session Preparation（Session Coordinator）
- 四类共享结构：
  - `MemoryBank`
  - `QuestionBank`
  - `SessionAgenda`
  - `BiographyDocument`
- 去重机制：QuestionBank 相似检索 `topK=3`
- 写作触发：未处理记忆达到 `>=10` 或用户结束会话

## 运行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://127.0.0.1:8080`

## 配置
编辑 `config/agents.yaml`：
- `openai.api_key`（建议改用环境变量 `OPENAI_API_KEY`）
- `models.*`（五个 agent 各自模型）
- `prompts.*`（五个 agent 提示词）

## 接口
- `POST /chat/start`：新建会话并初始化 Session Agenda
- `POST /chat/message`：处理用户输入（包含 Scribe 后台处理 + Interviewer 提问）
- `POST /chat/finish`：强制结束并完成剩余记忆写作

## 关键实现位置
- `agents/framework.py`：五 agent + 四数据结构 + 写作触发逻辑
- `app.py`：会话路由、状态管理与阶段编排
- `config/agents.yaml`：模型与提示词
