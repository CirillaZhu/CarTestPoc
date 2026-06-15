# -*- coding: utf-8 -*-
"""
最小 RAG - 网页测试界面 (Gradio)。
用法：
    $env:DEEPSEEK_API_KEY='sk-xxx'     # 可选，也可在网页里粘贴
    python app.py
然后浏览器打开  http://127.0.0.1:7860
"""
import os
import gradio as gr
import rag_core as core

# 服务端是否已配置 key（创空间环境变量）。配了就隐藏输入框，访客无需输入。
HAS_SERVER_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))


def ask(query, api_key, k):
    query = (query or "").strip()
    if not query:
        return "请输入问题。", [], ""
    hits = core.retrieve(query, int(k))
    answer = core.answer_with_deepseek(query, core.build_context(hits), api_key or None)
    if answer is None:
        answer = "_（未提供 DeepSeek key，下面仅为检索结果）_"
    gallery = [(core.media_path(m), f"[{m['type']}] {m['caption'] or '无标题'}")
               for m in core.media_of(hits) if os.path.exists(core.media_path(m))]
    sources = "\n".join(
        f"- **[{s:.3f}]** 《{c['doc']}》第{(c['page'] or 0)+1}页 / {c['heading'] or '正文'}"
        for c, s in hits
    )
    return answer, gallery, sources


with gr.Blocks(title="汽车标准智能问答 (RAG Demo)") as demo:
    gr.Markdown("## 汽车标准智能问答 · RAG Demo\n基于 MinerU 解析 + bge 本地向量检索 + DeepSeek 生成")
    with gr.Row():
        with gr.Column(scale=3):
            q = gr.Textbox(label="问题", placeholder="例如：F3级前雾灯的明暗截止线有什么要求？", lines=2)
        with gr.Column(scale=2):
            # 服务端已配 key 时隐藏此框，且永不预填真实 key（避免泄露到浏览器）
            key = gr.Textbox(label="DeepSeek API Key", type="password", value="",
                             visible=not HAS_SERVER_KEY,
                             placeholder="sk-...（留空则只做检索）")
            k = gr.Slider(1, 10, value=5, step=1, label="检索条数 Top-K")
    btn = gr.Button("提问", variant="primary")
    ans = gr.Markdown(label="答案")
    gr.Markdown("### 关联图片 / 表格")
    gallery = gr.Gallery(label="", columns=2, height=420, object_fit="contain")
    gr.Markdown("### 检索来源")
    src = gr.Markdown()

    btn.click(ask, inputs=[q, key, k], outputs=[ans, gallery, src])
    q.submit(ask, inputs=[q, key, k], outputs=[ans, gallery, src])


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE") == "1"        # 设为1则生成公网分享链接
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    # 云端(创空间)需绑 0.0.0.0 才能被代理访问；本地默认也可，bat 里会设回 127.0.0.1
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.launch(server_name=host, server_port=port,
                allowed_paths=[core.MEDIA_DIR], inbrowser=False, share=share)
