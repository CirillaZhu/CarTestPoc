# -*- coding: utf-8 -*-
"""RAG 核心逻辑，被 ask.py(命令行) 和 app.py(网页) 共用。"""
import os, json, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "index")
MEDIA_DIR = os.environ.get("RAG_MEDIA_DIR", os.path.join(HERE, "media"))  # 图片统一放这
# 本地已缓存的 ModelScope 模型目录（仅本机有；云端不存在会自动跳过）
_LOCAL_MODEL = r"D:\AI\models\modelscope\models\AI-ModelScope\bge-small-zh-v1___5"
_LOCAL_RERANKER = r"D:\AI\models\modelscope\models\BAAI\bge-reranker-base"
MODELSCOPE_ID = "AI-ModelScope/bge-small-zh-v1.5"
RERANKER_ID = "BAAI/bge-reranker-base"
QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："   # bge-zh 检索侧推荐前缀
TOP_K = 5
RERANK_TOPN = 20          # 一级召回候选数（送进 reranker 精排）
RERANK_THRESHOLD = 0.30   # 精排相关概率阈值，低于则判为不相关、丢弃（解决过召）
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _resolve_dir(env_key, local_dir, ms_id):
    """模型目录：环境变量 > 本地缓存 > ModelScope 下载。一份代码本地/云端通用。"""
    p = os.environ.get(env_key)
    if p and os.path.isdir(p):
        return p
    if os.path.isdir(local_dir):
        return local_dir
    from modelscope import snapshot_download
    return snapshot_download(ms_id)


@functools.lru_cache(maxsize=1)
def get_model():
    """embedding 模型（一级召回）只加载一次。"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_resolve_dir("EMB_MODEL_PATH", _LOCAL_MODEL, MODELSCOPE_ID))


@functools.lru_cache(maxsize=1)
def get_reranker():
    """cross-encoder 精排模型只加载一次。"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_resolve_dir("RERANKER_PATH", _LOCAL_RERANKER, RERANKER_ID))


def media_path(m):
    """把媒体条目解析成本地绝对路径（索引里只存文件名，便于跨机器搬运）。"""
    fname = m.get("file") or os.path.basename(m.get("path", ""))
    return os.path.join(MEDIA_DIR, fname)


@functools.lru_cache(maxsize=1)
def load_index():
    chunks = json.load(open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8"))
    emb = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    return chunks, emb


def retrieve(query, k=TOP_K, rerank=True, threshold=RERANK_THRESHOLD):
    """两阶段检索：
    ① 召回——向量相似度取 Top-N 候选（宁多勿漏）。
    ② 精筛——cross-encoder 把 query 与每个候选拼一起打分（能读懂"B级≠F3"），
       作为相关性门控：低于阈值的判为不相关直接丢弃（解决过召：弱相关块及其图片不再带出）。
    顺序仍沿用一级相似度（精确条款已排在前），reranker 只负责"砍掉不相关"。
    返回 [(chunk, 相关概率)]。"""
    chunks, emb = load_index()
    q = get_model().encode([QUERY_PROMPT + query], normalize_embeddings=True)[0]
    scores = emb @ q                                  # 余弦相似度
    cand_idx = np.argsort(-scores)[: (RERANK_TOPN if rerank else k)]

    if not rerank:
        return [(chunks[i], float(scores[i])) for i in cand_idx]

    # CrossEncoder 自带 Sigmoid，predict 直接返回 0~1 相关概率
    probs = get_reranker().predict([[query, chunks[i]["text"]] for i in cand_idx])
    kept = [(chunks[i], float(p)) for i, p in zip(cand_idx, probs) if p >= threshold][:k]
    if not kept:                                      # 全被过滤则保底给相似度最高的一条
        i = cand_idx[0]
        kept = [(chunks[i], float(scores[i]))]
    return kept


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
