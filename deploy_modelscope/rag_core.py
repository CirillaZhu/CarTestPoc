# -*- coding: utf-8 -*-
"""RAG 核心逻辑，被 ask.py(命令行) 和 app.py(网页) 共用。"""
import os, json, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "index")
MEDIA_DIR = os.environ.get("RAG_MEDIA_DIR", os.path.join(HERE, "media"))  # 图片统一放这
# 本地已缓存的 ModelScope 模型目录（仅本机有；云端不存在会自动跳过）
_LOCAL_MODEL = r"D:\AI\models\modelscope\models\AI-ModelScope\bge-small-zh-v1___5"
MODELSCOPE_ID = "AI-ModelScope/bge-small-zh-v1.5"
QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："   # bge-zh 检索侧推荐前缀
TOP_K = 5
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def resolve_model_dir():
    """按优先级拿到 embedding 模型目录：环境变量 > 本地缓存 > ModelScope 下载。
    这样同一份代码：本机用已下好的模型，云端(创空间)首次启动自动从 ModelScope 拉。"""
    p = os.environ.get("EMB_MODEL_PATH")
    if p and os.path.isdir(p):
        return p
    if os.path.isdir(_LOCAL_MODEL):
        return _LOCAL_MODEL
    from modelscope import snapshot_download
    return snapshot_download(MODELSCOPE_ID)


@functools.lru_cache(maxsize=1)
def get_model():
    """embedding 模型只加载一次。"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(resolve_model_dir())


def media_path(m):
    """把媒体条目解析成本地绝对路径（索引里只存文件名，便于跨机器搬运）。"""
    fname = m.get("file") or os.path.basename(m.get("path", ""))
    return os.path.join(MEDIA_DIR, fname)


@functools.lru_cache(maxsize=1)
def load_index():
    chunks = json.load(open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8"))
    emb = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    return chunks, emb


def retrieve(query, k=TOP_K):
    chunks, emb = load_index()
    q = get_model().encode([QUERY_PROMPT + query], normalize_embeddings=True)[0]
    scores = emb @ q                      # 已归一化，点积=余弦
    idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


def build_context(hits):
    parts = []
    for c, s in hits:
        src = f"《{c['doc']}》第{(c['page'] or 0)+1}页 / {c['heading'] or '正文'}"
        parts.append(f"[来源: {src}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_with_deepseek(query, context, api_key=None):
    from openai import OpenAI
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE)
    sys_prompt = (
        "你是汽车标准问答助手。只依据【参考资料】回答用户问题，"
        "不要编造；若资料不足请明说。回答末尾用「依据：」列出引用到的来源。"
    )
    user = f"【参考资料】\n{context}\n\n【问题】\n{query}"
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def media_of(hits):
    """收集命中块关联的图片/表格。"""
    out = []
    for c, _ in hits:
        out.extend(c.get("media", []))
    return out
