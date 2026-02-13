# 家传 Web Demo（Flask + 对话式 Multi-Agent）

这是一个基于 Flask 的网页 Demo，用于演示“对话式人物传记采访与写作”流程。

## 功能
- 对话采访代理：与用户多轮交互，逐步采集素材（非问卷）
- 充足性判断代理：判断信息是否足够写作
- 编排代理：输出章节结构
- 写作代理：生成传记初稿
- 配置驱动：通过 `config/agents.yaml` 配置 OpenAI 参数、模型和提示词

## 运行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://127.0.0.1:5000`

## 配置
编辑 `config/agents.yaml`：
- `openai.api_key`（建议改用环境变量 `OPENAI_API_KEY`）
- `openai.base_url`
- `models.*`（不同代理使用的模型）
- `prompts.*`（各代理提示词）

## 接口
- `POST /chat/start`：开启会话，返回欢迎语
- `POST /chat/message`：发送一轮用户输入，返回助手回复；当素材充足时自动返回结构与初稿


## 项目结构
- `app.py`：仅保留 Flask 路由与会话管理
- `agents/framework.py`：Multi-Agent 框架与 OpenAI 调用实现
- `config/agents.yaml`：模型、提示词与 API 配置
