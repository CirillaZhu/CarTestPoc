# -*- coding: utf-8 -*-
"""
最小 RAG - 网页测试界面 (Gradio)。
两个页签：
  · 问答         —— 原有的检索 + DeepSeek 生成
  · 知识库地图   —— 把每个知识块投到 2D，演示「新增文档」的过程与效果
用法：
    $env:DEEPSEEK_API_KEY='sk-xxx'     # 可选，也可在网页里粘贴
    python app.py
然后浏览器打开  http://127.0.0.1:7860
"""
import os
import gradio as gr
import plotly.graph_objects as go
import rag_core as core

# 服务端是否已配置 key（创空间环境变量）。配了就隐藏输入框，访客无需输入。
HAS_SERVER_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))

# 全部文档及各自块数（建库时即固定，用于地图页签的「已入库」勾选项）
_CHUNKS, _ = core.load_index()
ALL_DOCS = sorted({c["doc"] for c in _CHUNKS})
DOC_COUNT = {d: sum(1 for c in _CHUNKS if c["doc"] == d) for d in ALL_DOCS}
# 给每个文档分配一个稳定颜色
_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
DOC_COLOR = {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(ALL_DOCS)}


# ----------------------------- 问答页签 -----------------------------
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


# --------------------------- 知识库地图页签 ---------------------------
def _hover(c):
    """散点悬浮文字：来源 + 标题 + 正文片段。"""
    head = c["heading"] or "正文"
    snip = (c["text"] or "").replace("\n", " ")[:80]
    return f"《{c['doc']}》第{(c['page'] or 0)+1}页<br>{head}<br>{snip}…"


def render_map(active_docs, query, k):
    """画地图 + 返回库统计与检索对比文字。
    已入库文档=实色点簇；未入库=灰色空心点（预览新增后会落在哪片区域）。"""
    active = list(active_docs or [])
    chunks, emb = core.load_index()
    xy = core.project_points(emb)                       # 所有块的 2D 坐标（稳定）
    fig = go.Figure()
    for doc in ALL_DOCS:
        idx = [i for i, c in enumerate(chunks) if c["doc"] == doc]
        on = doc in active
        fig.add_trace(go.Scatter(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers",
            name=f"{doc} · {'已入库' if on else '未入库'} ({len(idx)})",
            text=[_hover(chunks[i]) for i in idx], hoverinfo="text",
            marker=dict(
                size=8 if on else 7,
                color=DOC_COLOR[doc] if on else "#c7ccd1",
                opacity=0.85 if on else 0.25,
                symbol="circle" if on else "circle-open",
                line=dict(width=0.5, color="white" if on else "#9aa0a6"),
            ),
        ))

    # 把问题投到同一空间，并只在「已入库」文档里检索
    hits_md = "_在上方输入问题，可把它投到地图上，并对比检索结果。_"
    query = (query or "").strip()
    if query:
        hits, qv = core.retrieve_among(query, active, int(k))
        qxy = core.project_points(qv)[0]
        fig.add_trace(go.Scatter(
            x=[qxy[0]], y=[qxy[1]], mode="markers", name="❓ 你的问题",
            marker=dict(size=20, color="#111827", symbol="star",
                        line=dict(width=1, color="white")),
            hoverinfo="text", text=[f"问题：{query}"],
        ))
        # 从问题连线到命中的块，直观显示「答案从哪来」
        for c, s in hits:
            ci = next(i for i, x in enumerate(chunks)
                      if x["doc"] == c["doc"] and x["id"] == c["id"])
            fig.add_trace(go.Scatter(
                x=[qxy[0], xy[ci, 0]], y=[qxy[1], xy[ci, 1]], mode="lines",
                line=dict(width=1, color="#111827", dash="dot"),
                opacity=0.4, hoverinfo="skip", showlegend=False,
            ))
        if hits:
            best = hits[0]
            hits_md = (f"**问题落点最近的 {len(hits)} 个知识块**"
                       f"（仅在已入库文档中检索，最高相似度 **{best[1]:.3f}**，"
                       f"来自《{best[0]['doc']}》）：\n\n" + "\n".join(
                f"- **[{s:.3f}]** 《{c['doc']}》第{(c['page'] or 0)+1}页 / {c['heading'] or '正文'}"
                for c, s in hits))
        else:
            hits_md = "**当前知识库为空**（未勾选任何文档），无法检索。勾选文档即完成「新增入库」。"

    n_chunk = sum(DOC_COUNT[d] for d in active)
    total = sum(DOC_COUNT.values())
    stats_md = (f"### 当前知识库：**{len(active)} / {len(ALL_DOCS)}** 篇文档 · "
                f"**{n_chunk} / {total}** 个知识块\n"
                "勾选 = 该文档已新增入库（实色点簇）；取消 = 尚未入库（灰色预览）。"
                "对同一个问题勾选 / 取消文档，观察落点与检索结果的变化，即是「新增知识的过程与效果」。")

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig, stats_md, hits_md


# ------------------------------- 界面 -------------------------------
with gr.Blocks(title="汽车标准智能问答 (RAG Demo)") as demo:
    gr.Markdown("## 汽车标准智能问答 · RAG Demo\n基于 MinerU 解析 + bge 本地向量检索 + DeepSeek 生成")

    with gr.Tab("问答"):
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

    with gr.Tab("知识库地图"):
        gr.Markdown(
            "每个点 = 一个知识块，颜色 = 来源文档；语义相近的块在图上彼此靠近。\n"
            "**演示新增**：取消勾选某文档=它尚未入库（灰色点预览覆盖区域），勾选=完成新增入库。"
        )
        with gr.Row():
            docs_cb = gr.CheckboxGroup(
                choices=ALL_DOCS, value=ALL_DOCS, label="已入库文档（勾选即新增）")
        with gr.Row():
            mq = gr.Textbox(scale=4, label="把问题投到地图上（可选）",
                            placeholder="例如：F3级前雾灯的明暗截止线有什么要求？")
            mk = gr.Slider(1, 10, value=5, step=1, label="高亮命中数")
        map_stats = gr.Markdown()
        map_plot = gr.Plot()
        map_hits = gr.Markdown()

        _inputs = [docs_cb, mq, mk]
        _outputs = [map_plot, map_stats, map_hits]
        for comp in (docs_cb, mq, mk):
            comp.change(render_map, inputs=_inputs, outputs=_outputs)
        mq.submit(render_map, inputs=_inputs, outputs=_outputs)
        demo.load(render_map, inputs=_inputs, outputs=_outputs)


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE") == "1"        # 设为1则生成公网分享链接
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    # 云端(创空间)需绑 0.0.0.0 才能被代理访问；本地默认也可，bat 里会设回 127.0.0.1
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.launch(server_name=host, server_port=port,
                allowed_paths=[core.MEDIA_DIR], inbrowser=False, share=share)
