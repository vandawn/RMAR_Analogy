# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from collections import defaultdict
import random

# ---------------- 路径配置（按你的机器已对齐） ----------------
ROOT = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME"
DATA_DIR = os.path.join(ROOT, "data/MCNet")

ENT_ID = os.path.join(DATA_DIR, "ent_id")
REL_ID = os.path.join(DATA_DIR, "rel_id")

# 三元组来源：优先读 pickle；否则读 txt
TRI_PKL_1 = os.path.join(DATA_DIR, "wiki_tuple_ids.pickle")  # (N,3) uint64
TRI_TXT_1 = os.path.join(ROOT, "src_data/MCNet/wiki_tuple_ids")  # 每行: lhs \t rel \t rhs
# 也兼容 train/valid/test.pickle 三件套（若存在）
TRAIN_PKL = os.path.join(DATA_DIR, "train.pickle")
VALID_PKL = os.path.join(DATA_DIR, "valid.pickle")
TEST_PKL  = os.path.join(DATA_DIR, "test.pickle")

# 实体 → 相对图片路径
BEST_IMG = os.path.join(DATA_DIR, "analogy_best_img.pickle")
# 图片根目录
IMG_ROOT = "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images"
# 图片向量：{绝对图片路径: np.array(1000,)}
IMG_VEC_DICT = os.path.join(DATA_DIR, "analogy_vit_best_img_vec.pickle")

# 输出
OUT_RANK = os.path.join(DATA_DIR, "analogy_vit_rank.txt")
# -----------------------------------------------------------

SEED = 42
NEG_K = 100      # 每条采样的负样本数
COSINE_EPS = 1e-8


def load_ent_rel():
    ent2id = {}
    with open(ENT_ID, "r", encoding="utf-8") as f:
        for ln in f:
            k, v = ln.rstrip("\n").split("\t")
            ent2id[k] = int(v)
    id2ent = {v: k for k, v in ent2id.items()}

    rel2id = {}
    with open(REL_ID, "r", encoding="utf-8") as f:
        for ln in f:
            k, v = ln.rstrip("\n").split("\t")
            rel2id[k] = int(v)
    id2rel = {v: k for k, v in rel2id.items()}

    return ent2id, id2ent, rel2id, id2rel


def load_triples(ent2id, rel2id):
    # 1) 尝试读 (N,3) 的 pickle
    if os.path.exists(TRI_PKL_1):
        arr = pickle.load(open(TRI_PKL_1, "rb"))
        assert arr.ndim == 2 and arr.shape[1] >= 3
        return arr[:, :3].astype(np.int64)

    # 2) train/valid/test.pickle 三件套
    # picks = []
    # for p in [TRAIN_PKL, VALID_PKL, TEST_PKL]:
    #     if os.path.exists(p):
    #         a = pickle.load(open(p, "rb"))
    #         picks.append(a[:, :3].astype(np.int64))
    # if picks:
    #     return np.concatenate(picks, axis=0)
    if os.path.exists(TRAIN_PKL):
        a = pickle.load(open(TRAIN_PKL, "rb"))
        return a[:, :3].astype(np.int64)

    # 3) txt（字符串三元组，需要映射）
    if os.path.exists(TRI_TXT_1):
        rows = []
        with open(TRI_TXT_1, "r", encoding="utf-8") as f:
            for ln in f:
                h, r, t = ln.rstrip("\n").split("\t")
                if h in ent2id and r in rel2id and t in ent2id:
                    rows.append([ent2id[h], rel2id[r], ent2id[t]])
        return np.array(rows, dtype=np.int64)

    raise FileNotFoundError(
        "No triples found. Expected one of:\n"
        f"  {TRI_PKL_1}\n  {TRAIN_PKL},{VALID_PKL},{TEST_PKL}\n  {TRI_TXT_1}"
    )


def build_entity_vectors():
    # {entity -> relpath}
    ent2rel = pickle.load(open(BEST_IMG, "rb"))
    # {abspath -> vec}
    path2vec = pickle.load(open(IMG_VEC_DICT, "rb"))

    # 把每个实体映射到其图片向量（没有则 None）
    ent2vec = {}
    dim = None
    miss = 0
    for ent, relp in ent2rel.items():
        absp = os.path.join(IMG_ROOT, relp)
        vec = path2vec.get(absp)
        if vec is None:
            ent2vec[ent] = None
            miss += 1
        else:
            v = np.asarray(vec, dtype=np.float32)
            if dim is None:
                dim = v.size
            ent2vec[ent] = v
    print(f"[vectors] mapped entities: {len(ent2vec)}  missing: {miss}")
    return ent2vec, dim or 1000


def cosine(a, b):
    # a: (D,), b: (M,D) or (D,)
    if b.ndim == 1:
        num = (a * b).sum()
        den = (np.linalg.norm(a) + COSINE_EPS) * (np.linalg.norm(b) + COSINE_EPS)
        return float(num / den)
    else:
        num = b @ a
        den = (np.linalg.norm(a) + COSINE_EPS) * (np.linalg.norm(b, axis=1) + COSINE_EPS)
        return num / den


def compute_rank_per_relation():
    random.seed(SEED)
    np.random.seed(SEED)

    ent2id, id2ent, rel2id, id2rel = load_ent_rel()
    triples = load_triples(ent2id, rel2id)  # (N,3): h,r,t (id)
    ent2vec, dim = build_entity_vectors()

    # 有向量的实体ID集合
    has_vec_id = set()
    for ent, idx in ent2id.items():
        v = ent2vec.get(ent)
        if v is not None and np.linalg.norm(v) > 0:
            has_vec_id.add(idx)
    if len(has_vec_id) < 3:
        raise RuntimeError("Too few entities with image vectors.")

    # 预分组
    by_rel = defaultdict(list)
    for h, r, t in triples:
        by_rel[r].append((h, t))

    lines = []
    for rid, pairs in by_rel.items():
        total = len(pairs)

        # 过滤两端都有向量的
        usable = [(h, t) for (h, t) in pairs if h in has_vec_id and t in has_vec_id]
        used = len(usable)
        if used == 0:
            rank_mean = 1.0  # or np.nan
            lines.append(f"relation id: {id2rel[rid]}\trank: {rank_mean}\tpercentage: 0.0/{total}")
            continue

        # 为了效率，准备一个“有向量的实体ID数组”
        ent_pool = np.array(sorted(has_vec_id), dtype=np.int64)

        # 构建 id->向量（单位向量，cosine 更稳）
        vec_cache = {}
        for eid in has_vec_id:
            ent_str = id2ent[eid]
            v = ent2vec[ent_str].astype(np.float32)
            n = np.linalg.norm(v) + COSINE_EPS
            vec_cache[eid] = v / n

        ranks = []
        for h, t in usable:
            vh = vec_cache[h]
            vt = vec_cache[t]
            sim_true = float(np.dot(vh, vt))  # 归一化后点积=cosine

            # 负样本（tail 侧）：从有向量实体里采样不同于 t 的 NEG_K 个
            # （也可以做 head 侧，或两侧都做后取平均——按需可扩展）
            cand = np.random.choice(ent_pool, size=min(NEG_K, len(ent_pool)-1), replace=False)
            cand = cand[cand != t]
            if cand.size == 0:
                continue
            sims_neg = np.array([np.dot(vh, vec_cache[neg]) for neg in cand], dtype=np.float32)

            # 百分位名次：越小越好；= (#neg >= true) / #neg
            rank = float((sims_neg >= sim_true).sum()) / float(cand.size)
            ranks.append(rank)

        rank_mean = float(np.mean(ranks)) if ranks else 1.0
        rel_str = id2rel[rid]
        lines.append(f"relation id: {rel_str}\trank: {rank_mean}\tpercentage: {float(used)}/{total}")

    with open(OUT_RANK, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"[OK] wrote {OUT_RANK} ({len(lines)} relations)")
    return OUT_RANK


if __name__ == "__main__":
    compute_rank_per_relation()
