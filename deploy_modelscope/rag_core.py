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


# ---------------------------------------------------------------------------
# 知识库地图：把高维向量降到 2D 看「覆盖面」，并把问题投到同一空间看检索效果。
# 降维用纯 numpy 的 PCA（SVD），不引入 sklearn/umap，云端零额外重依赖。
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _pca_basis():
    """在全部向量上拟合一次 PCA 主轴：返回 (均值, 前两主成分)。
    全量拟合保证坐标稳定——某文档是否「已入库」只影响显示，不会让点乱跳。"""
    _, emb = load_index()
    mean = emb.mean(axis=0)
    _, _, vt = np.linalg.svd(emb - mean, full_matrices=False)
    return mean, vt[:2]                     # comps: (2, d)


def project_points(vectors):
    """把任意向量（含库内块或新问题）投到同一张 2D 地图上。"""
    mean, comps = _pca_basis()
    vectors = np.asarray(vectors)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    return (vectors - mean) @ comps.T       # (n, 2)


def embed_query(query):
    """对问题做检索侧向量化（带 bge 前缀、已归一化）。"""
    return get_model().encode([QUERY_PROMPT + query], normalize_embeddings=True)[0]


def retrieve_among(query, active_docs, k=TOP_K):
    """只在「已入库」的文档集合里检索，用于演示新增前后的检索差异。
    返回 (hits, 问题向量)，问题向量供地图投点复用。"""
    chunks, emb = load_index()
    qv = embed_query(query)
    scores = emb @ qv
    active = set(active_docs or [])
    out = []
    for i in np.argsort(-scores):
        if chunks[i]["doc"] in active:
            out.append((chunks[i], float(scores[i])))
            if len(out) >= k:
                break
    return out, qv
