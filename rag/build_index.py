# -*- coding: utf-8 -*-
"""
最小 RAG - 步骤1：把 MinerU 的 content_list.json 切块、向量化、存索引。
用法：  python build_index.py
产出：  rag/index/chunks.json  +  rag/index/embeddings.npy
"""
import os, sys, json, glob, io, shutil
import numpy as np
import rag_core as core          # 模型路径、索引目录等配置单一来源

# 控制台强制 UTF-8，避免 Windows GBK 乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(core.HERE)
MINERU_OUT = os.path.join(ROOT, "mineru_output")
INDEX_DIR = core.INDEX_DIR
MEDIA_DIR = core.MEDIA_DIR
MAX_CHARS = 700          # 单块文本上限，超过就切
SKIP = {"header", "footer", "page_number"}

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)


def flatten_table(html: str) -> str:
    """把表格 HTML 粗略转成可检索的纯文本。"""
    import re
    txt = re.sub(r"</td>", " | ", html)
    txt = re.sub(r"</tr>", "\n", txt)
    txt = re.sub(r"<[^>]+>", "", txt)
    return re.sub(r"[ \t]+", " ", txt).strip()


def chunk_document(auto_dir: str, doc: str):
    cl_path = glob.glob(os.path.join(auto_dir, "*_content_list.json"))[0]
    blocks = json.load(open(cl_path, encoding="utf-8"))

    chunks, buf, media, heading, page = [], [], [], "", None

    def flush():
        nonlocal buf, media
        text = "\n".join(buf).strip()
        if text or media:
            chunks.append({
                "id": f"{doc}#{len(chunks)}",
                "doc": doc,
                "page": page,           # 0-based 页码
                "heading": heading,
                "text": text,
                "media": media,
            })
        buf, media = [], []

    for b in blocks:
        t = b.get("type")
        if t in SKIP:
            continue
        if page is None:
            page = b.get("page_idx")

        if t == "text":
            txt = (b.get("text") or "").strip()
            if not txt:
                continue
            # 有 text_level 视为标题：作为新块边界
            if b.get("text_level"):
                flush()
                heading = txt
                page = b.get("page_idx")
                buf.append(txt)
            else:
                buf.append(txt)
                if sum(len(x) for x in buf) > MAX_CHARS:
                    flush()
                    page = b.get("page_idx")
        elif t == "equation":
            buf.append(b.get("text", ""))          # LaTeX 公式
        elif t == "page_footnote":
            buf.append("[注] " + (b.get("text") or ""))
        elif t in ("image", "chart", "table"):
            cap = " ".join(b.get(f"{ 'table' if t=='table' else 'chart' if t=='chart' else 'image' }_caption", []) or [])
            img = b.get("img_path")
            if img:
                src = os.path.normpath(os.path.join(auto_dir, img))
                fname = os.path.basename(src)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(MEDIA_DIR, fname))  # 汇集到 media/
                media.append({
                    "type": t,
                    "file": fname,          # 只存文件名，可移植
                    "caption": cap,
                })
            if t == "table" and b.get("table_body"):
                buf.append((cap + "\n" if cap else "") + flatten_table(b["table_body"]))
            elif cap:
                buf.append(f"[{t}] {cap}")
    flush()
    return chunks


def main():
    all_chunks = []
    for auto_dir in glob.glob(os.path.join(MINERU_OUT, "*", "auto")):
        doc = os.path.basename(os.path.dirname(auto_dir))
        cs = chunk_document(auto_dir, doc)
        print(f"  {doc}: {len(cs)} 块")
        all_chunks.extend(cs)
    print(f"总计 {len(all_chunks)} 块，开始向量化...")

    model = core.get_model()      # 与检索端用同一个模型（同一来源）
    texts = [f"{c['heading']}\n{c['text']}".strip() for c in all_chunks]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), emb)
    json.dump(all_chunks, open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"已保存索引到 {INDEX_DIR}  (向量维度 {emb.shape})")


if __name__ == "__main__":
    main()
