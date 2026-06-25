# -*- coding: utf-8 -*-
"""
最小 RAG - 步骤1：把 MinerU 的 content_list.json 切块、向量化、存索引。
用法：  python build_index.py
产出：  rag/index/chunks.json  +  rag/index/embeddings.npy
"""
import os, sys, json, glob, io, shutil, re
import numpy as np
import rag_core as core          # 模型路径、索引目录等配置单一来源

# 条款编号：4 / 5.9 / 5.9.3.1 …（后面必须跟空白再接内容）
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\s+|　)(.*)$", re.S)

# 控制台强制 UTF-8，避免 Windows GBK 乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(core.HERE)
MINERU_OUT = os.path.join(ROOT, "mineru_output")
INDEX_DIR = core.INDEX_DIR
MEDIA_DIR = core.MEDIA_DIR
SKIP = {"header", "footer", "page_number"}

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)


def flatten_table(html: str) -> str:
    """把表格 HTML 粗略转成可检索的纯文本。"""
    txt = re.sub(r"</td>", " | ", html)
    txt = re.sub(r"</tr>", "\n", txt)
    txt = re.sub(r"<[^>]+>", "", txt)
    return re.sub(r"[ \t]+", " ", txt).strip()


def is_clause(num: str) -> bool:
    """形如 4 / 5.9 / 5.9.3.1，段数≤6 且首段为合理章号（排除 175cd 这类误判）。"""
    parts = num.split(".")
    return all(p.isdigit() for p in parts) and 1 <= len(parts) <= 6 and int(parts[0]) < 100


def caption_of(b: dict, t: str) -> str:
    key = "table" if t == "table" else "chart" if t == "chart" else "image"
    return " ".join(b.get(f"{key}_caption", []) or [])


def copy_media(auto_dir: str, img: str) -> str:
    src = os.path.normpath(os.path.join(auto_dir, img))
    fname = os.path.basename(src)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(MEDIA_DIR, fname))
    return fname


def breadcrumb(num: str, titles: dict) -> str:
    """5.9.3.1 -> '5 技术要求 > 5.9 … > 5.9.3 F3级前雾灯配光性能要求 > 5.9.3.1'，给叶子补父级上下文。"""
    parts = num.split(".")
    crumbs = []
    for i in range(1, len(parts) + 1):
        anc = ".".join(parts[:i])
        crumbs.append(f"{anc} {titles.get(anc, '')}".strip())
    return " > ".join(crumbs)


def chunk_document(auto_dir: str, doc: str):
    cl_path = glob.glob(os.path.join(auto_dir, "*_content_list.json"))[0]
    blocks = json.load(open(cl_path, encoding="utf-8"))

    nodes, order, titles, cur = {}, [], {}, None  # 编号->节点；保序；编号->标题；当前节点

    def node(key, page):
        if key not in nodes:
            nodes[key] = {"body": [], "media": [], "page": page}
            order.append(key)
        return nodes[key]

    for b in blocks:
        t = b.get("type")
        if t in SKIP:
            continue
        page = b.get("page_idx")

        if t == "text":
            txt = (b.get("text") or "").strip()
            if not txt:
                continue
            m = CLAUSE_RE.match(txt)
            if m and is_clause(m.group(1)):
                num, rest = m.group(1), m.group(2).strip()
                n = node(num, page)
                if b.get("text_level"):            # 编号标题（5.9.3 …）：只记标题进面包屑
                    titles[num] = rest
                else:                              # 编号条款正文 = 叶子块
                    n["body"].append(rest or txt)
                cur = num
            elif b.get("text_level"):              # 无编号标题：前言/范围/附录
                cur = "§" + txt
                node(cur, page)
                titles[cur] = txt
            else:                                  # 无编号正文 → 并入当前条款
                cur = cur or "§前置"
                node(cur, page)["body"].append(txt)
        elif t == "equation":
            cur = cur or "§前置"
            node(cur, page)["body"].append(b.get("text", ""))
        elif t == "page_footnote":
            if cur:
                nodes[cur]["body"].append("[注] " + (b.get("text") or ""))
        elif t in ("image", "chart", "table"):
            cur = cur or "§前置"
            n = node(cur, page)
            cap = caption_of(b, t)
            if b.get("img_path"):
                n["media"].append({"type": t, "file": copy_media(auto_dir, b["img_path"]), "caption": cap})
            if t == "table" and b.get("table_body"):
                n["body"].append((cap + "\n" if cap else "") + flatten_table(b["table_body"]))
            elif cap:
                n["body"].append(f"[{t}] {cap}")

    chunks = []
    for key in order:
        n = nodes[key]
        body = "\n".join(x for x in n["body"] if x).strip()
        if not body and not n["media"]:            # 纯标题节点跳过（信息已在子块面包屑里）
            continue
        is_num = not key.startswith("§")
        chunks.append({
            "id": f"{doc}#{key}",
            "doc": doc,
            "page": n["page"],
            "clause": key if is_num else "",
            "heading": breadcrumb(key, titles) if is_num else key[1:],
            "text": body,
            "media": n["media"],
        })
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
    # 只把条款正文入库；面包屑(heading)仅用于展示/引用，放进向量会稀释判别词
    texts = [c["text"] for c in all_chunks]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), emb)
    json.dump(all_chunks, open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"已保存索引到 {INDEX_DIR}  (向量维度 {emb.shape})")


if __name__ == "__main__":
    main()
