# 家传 Web Demo（Flask + Multi-Agent）

这是一个基于 Flask + HTML 的网页 Demo，用于演示“自动化人物传记采访与写作”流程。

## 功能
- 采访代理：生成结构化采访提纲并提炼输入素材
- 编排代理：输出章节框架
- 写作代理：生成传记初稿

## 运行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://127.0.0.1:5000`

## 接口
`POST /generate`
- 表单字段：`subject_name`, `era`, `highlights`, `raw_notes`, `tone`, `generation_goal`
- 设置 `Accept: application/json` 可返回 JSON。
