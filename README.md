# ST-research-proof

ST-research-proof 是一个本地优先的证据定位型 AI 检索系统，旨在解决普通 AI 搜索中“答案难以精确验证、引用粒度过粗、私有资料不便上传”的问题。系统支持本地知识库、网页搜索、离线检索和可切换模型后端，并将本地文档与网页内容统一抽象为 evidence 证据对象。每条证据包含来源、位置、原文摘录和证据编号，使模型回答能够绑定到具体文档段落或网页正文段落，而不仅仅停留在来源链接层面。

本项目更关注可信问答、证据溯源和结果可复核性，适用于课程资料、科研笔记、政策文档、本地知识库、题库资料和网页信息查证等场景。

## 核心功能

- 本地知识库：支持将文档放入 `kb/documents/` 后自动索引。
- 离线检索：可在 `SEARCH_MODE=LOCAL` 模式下仅检索本地资料。
- 证据定位：本地文档支持段落级定位，网页结果支持正文段落定位和原文摘录。
- 证据编号：回答中的关键结论可以绑定 `[L1]`、`[E1]` 等证据编号。
- 题库直抽：对包含“正确答案:”等结构化字段的资料，可直接抽取答案，减少模型幻觉和生成耗时。
- 网页证据链：网页搜索结果会尝试抽取正文段落，并提供 URL、段落位置、原文摘录和定位链接。
- 可切换模型后端：支持本地 Transformers 模型，也支持 OpenAI-compatible 模型服务，例如 llama.cpp、Ollama、LM Studio 等。
- 可选联网搜索：可通过 SearXNG 进行联网搜索，也可以完全关闭联网功能。

## 项目结构

ST-research-proof/
├── run.py                  # 后端主程序
├── index.html              # 前端页面
├── cfg.txt                 # 配置文件
├── start.bat               # Windows 启动脚本
├── local_kb.py             # 本地知识库索引与检索
├── web_evidence.py         # 网页证据抽取与定位
├── direct_sources.py       # GitHub / URL 等直连来源解析
├── model_backends.py       # 模型后端抽象
├── answer_extractors.py    # 题库类答案直抽
├── settings.yml            # SearXNG 配置
├── website.txt             # 网站权重配置
├── skill/
│   └── search.txt          # 搜索与回答规则
└── kb/
    ├── documents/          # 本地知识库原始文件
    └── kb.sqlite           # 自动生成的本地索引数据库
