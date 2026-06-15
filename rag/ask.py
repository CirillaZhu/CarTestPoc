# -*- coding: utf-8 -*-
"""
最小 RAG - 命令行问答入口。检索/向量化/调 DeepSeek 全部复用 rag_core。
用法：
    set DEEPSEEK_API_KEY=sk-xxx      (PowerShell: $env:DEEPSEEK_API_KEY='sk-xxx')
    python ask.py "F3级前雾灯的明暗截止线有什么要求？" [--html]
没有设置 key 时，只做检索、打印命中的内容（方便先验证检索效果）。
"""
import os, sys, io
import rag_core as core

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def render_html(query, answer, hits, media_all):
    """生成一个带内嵌图片的 HTML 报告，浏览器里能直接看到答案+图。"""
    import html, webbrowser, pathlib
    def esc(x): return html.escape(x or "")
    rows = ""
    for c, s in hits:
        rows += (f"<li><b>[{s:.3f}]</b> 《{esc(c['doc'])}》第{(c['page'] or 0)+1}页 / "
                 f"{esc(c['heading'] or '正文')}<br><span style='color:#555'>"
                 f"{esc(c['text'][:200])}</span></li>")
    imgs = ""
    for m in media_all:
        uri = pathlib.Path(core.media_path(m)).as_uri()
        imgs += (f"<figure style='margin:12px 0'><img src='{uri}' "
                 f"style='max-width:680px;border:1px solid #ccc'>"
                 f"<figcaption style='color:#666'>[{m['type']}] {esc(m['caption'])}</figcaption></figure>")
    doc = f"""<!doctype html><meta charset="utf-8">
<body style="font:15px/1.7 system-ui,'Microsoft YaHei';max-width:760px;margin:30px auto;padding:0 16px">
<h2>问题</h2><p>{esc(query)}</p>
<h2>答案（DeepSeek）</h2><div style="white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:8px">{esc(answer)}</div>
<h2>关联图片 / 表格</h2>{imgs or '<p>（无）</p>'}
<h2>检索来源</h2><ol>{rows}</ol></body>"""
    out = os.path.join(core.HERE, "last_answer.html")
    open(out, "w", encoding="utf-8").write(doc)
    webbrowser.open(pathlib.Path(out).as_uri())
    print(f"\n[已生成 HTML 报告并打开] {out}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_html = "--html" in sys.argv
    if not args:
        print('用法: python ask.py "你的问题" [--html]'); return
    query = args[0]
    hits = core.retrieve(query)

    print(f"\n问题：{query}\n")
    print("===== 检索命中（Top {}）=====".format(len(hits)))
    for c, s in hits:
        print(f"\n[{s:.3f}] 《{c['doc']}》第{(c['page'] or 0)+1}页 / {c['heading'] or '正文'}")
        print("  " + (c["text"][:160].replace("\n", " ") + ("..." if len(c["text"]) > 160 else "")))

    media_all = core.media_of(hits)
    if media_all:
        print("\n===== 关联图片/表格 =====")
        for m in media_all:
            print(f"  [{m['type']}] {m['caption'] or '(无标题)'}  ->  {core.media_path(m)}")

    answer = core.answer_with_deepseek(query, core.build_context(hits))
    if answer is not None:
        print("\n===== DeepSeek 回答 =====")
        print(answer)
    else:
        print("\n[提示] 未设置 DEEPSEEK_API_KEY，仅展示检索结果。"
              "设置后再运行即可得到大模型生成的答案。")

    if want_html:
        render_html(query, answer or "(未设置 key，无生成答案)", hits, media_all)


if __name__ == "__main__":
    main()
