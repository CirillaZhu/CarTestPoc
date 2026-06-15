---
license: Apache License 2.0
domain:
  - nlp
tags:
  - rag
  - qa
  - deepseek
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
---

# 汽车标准智能问答 (RAG Demo)

基于 **MinerU 解析 + bge-small-zh 本地向量检索 + DeepSeek 生成** 的最小 RAG 演示。
知识库为两份汽车国家标准（GB 4660-2016 前雾灯、GB 11551-2014）。

## 使用
1. 直接输入问题，例如「F3级前雾灯的明暗截止线有什么要求？」
2. 下方会给出答案、关联的图片/表格、以及检索来源。

> DeepSeek key 由创空间**环境变量**提供（见部署说明），访客无需输入。
> 若未配置环境变量，页面会出现 key 输入框，留空则只返回检索结果不生成答案。

## 说明
- embedding 模型 `bge-small-zh-v1.5` 首次启动时从 ModelScope 自动下载。
- 检索在 CPU 上即可秒级完成。
