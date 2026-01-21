#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from collections import defaultdict

# ===== 路径配置（按你的环境已对齐）=====
ROOT     = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME"
DATA_DIR = os.path.join(ROOT, "data/MCNet")

ENT_ID = os.path.join(DATA_DIR, "ent_id")
REL_ID = os.path.join(DATA_DIR, "rel_id")

# 三元组来源（三选一，按优先级自动取）
TRI_PKL_ALL  = os.path.join(DATA_DIR, "wiki_tuple_ids.pickle")        # (N,3)
TRAIN_PKL    = os.path.join(DATA_DIR, "train.pickle")                  # (N,3)
VALID_PKL    = os.path.join(DATA_DIR, "valid.pickle")                  # (N,3)
TEST_PKL     = os.path.join(DATA_DIR, "test.pickle")                   # (N,3)
TRI_TXT_RAW  = os.path.join(ROOT, "src_data/MCNet/wiki_tuple_ids")     # h \t r \t t (strings)

# 输出（保持与 analogy 同名）
OUT_TO_SKIP  = os.path.join(DATA_DIR, "to_skip.pickle")
OUT_PROBAS   = os.path.join(DATA_DIR, "probas.pickle")
OUT_1TO1     = os.path.join(DATA_DIR, "analogy_1_1_triples.pickle")
# =======================================


def load_id_map(path):
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or "\t" not in ln:
                continue
            k, v = ln.split("\t", 1)
            try:
                mp[k] = int(v)
            except ValueError:
                continue
    return mp

def id2key_from_map(mp):
    id2k = [None] * (max(mp.values()) + 1) if mp else []
    for k, i in mp.items():
        if i >= 0:
            if i >= len(id2k):
                id2k += [None] * (i - len(id2k) + 1)
            id2k[i] = k
    return id2k

def load_triples(ent2id, rel2id):
    # 1) 直接读全集 pickle
    if os.path.exists(TRI_PKL_ALL):
        arr = pickle.load(open(TRI_PKL_ALL, "rb"))
        arr = arr[:, :3].astype(np.uint64)
        return np.unique(arr, axis=0)

    # 2) 合并 train/valid/test（注意去重；你三份是同一份，不影响）
    picks = []
    for p in [TRAIN_PKL, VALID_PKL, TEST_PKL]:
        if os.path.exists(p):
            a = pickle.load(open(p, "rb"))
            picks.append(a[:, :3].astype(np.uint64))
    if picks:
        arr = np.concatenate(picks, axis=0)
        return np.unique(arr, axis=0)

    # 3) 从原始 txt 做映射
    if os.path.exists(TRI_TXT_RAW):
        rows = []
        with open(TRI_TXT_RAW, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln: continue
                parts = ln.split("\t")
                if len(parts) != 3: continue
                h, r, t = parts
                if h in ent2id and r in rel2id and t in ent2id:
                    rows.append([ent2id[h], rel2id[r], ent2id[t]])
        arr = np.array(rows, dtype=np.uint64)
        return np.unique(arr, axis=0)

    raise FileNotFoundError(
        "No triples found. Expected one of:\n"
        f"  {TRI_PKL_ALL}\n"
        f"  {TRAIN_PKL},{VALID_PKL},{TEST_PKL}\n"
        f"  {TRI_TXT_RAW}\n"
    )

def build_to_skip_and_probas(all_triples, n_rel, n_ent):
    # to_skip（用于 filtered 评测）
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for lhs, rel, rhs in all_triples.astype(np.int64):
        to_skip['lhs'][(rhs, rel + n_rel)].add(lhs)  # reciprocals
        to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    # probas（实体出现分布）
    counters = {
        'lhs': np.zeros(n_ent, dtype=np.float64),
        'rhs': np.zeros(n_ent, dtype=np.float64),
        'both': np.zeros(n_ent, dtype=np.float64),
    }
    for lhs, rel, rhs in all_triples.astype(np.int64):
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k in counters:
        s = counters[k].sum()
        counters[k] = counters[k] / s if s > 0 else counters[k]

    return to_skip_final, counters

def build_1to1_triples(all_triples, id2ent, id2rel):
    """
    返回 list[ (head_str, rel_str, tail_str) ]，满足：
      对同一关系 r，任一头仅指向 1 个尾，且任一尾仅来自 1 个头。
    """
    HRT = all_triples.astype(np.int64)
    # 统计 (r,h)->#tails, (r,t)->#heads
    rh_count = defaultdict(int)
    rt_count = defaultdict(int)
    for h, r, t in HRT:
        rh_count[(r, h)] += 1
        rt_count[(r, t)] += 1

    out = []
    for h, r, t in HRT:
        if rh_count[(r, h)] == 1 and rt_count[(r, t)] == 1:
            hs = id2ent[h] if 0 <= h < len(id2ent) else f"<E{h}>"
            rs = id2rel[r] if 0 <= r < len(id2rel) else f"<R{r}>"
            ts = id2ent[t] if 0 <= t < len(id2ent) else f"<E{t}>"
            out.append((hs, rs, ts))
    return out

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    ent2id = load_id_map(ENT_ID)
    rel2id = load_id_map(REL_ID)
    id2ent = id2key_from_map(ent2id)
    id2rel = id2key_from_map(rel2id)
    print(f"[load] entities={len(ent2id)}, relations={len(rel2id)}")

    triples = load_triples(ent2id, rel2id)   # (N,3) uint64 (已去重)
    print(f"[triples] total(unique)={len(triples)}  shape={triples.shape}  dtype={triples.dtype}")

    # to_skip / probas
    to_skip, probas = build_to_skip_and_probas(triples, n_rel=len(rel2id), n_ent=len(ent2id))
    with open(OUT_TO_SKIP, "wb") as f:
        pickle.dump(to_skip, f)
    with open(OUT_PROBAS, "wb") as f:
        pickle.dump(probas, f)
    print(f"[save] to_skip -> {OUT_TO_SKIP}")
    print(f"[save] probas  -> {OUT_PROBAS}")

    # 1-1 triples（与 analogy 格式一致：字符串三元组列表）
    triples_1to1 = build_1to1_triples(triples, id2ent=id2ent, id2rel=id2rel)
    with open(OUT_1TO1, "wb") as f:
        pickle.dump(triples_1to1, f)
    print(f"[save] analogy_1_1_triples -> {OUT_1TO1}  (size={len(triples_1to1)})")

    # 小样例（前 5 条）
    k = min(5, len(triples_1to1))
    if k:
        print("\n-- sample of analogy_1_1_triples --")
        for i in range(k):
            print(f"[{i+1}] {triples_1to1[i]}")

if __name__ == "__main__":
    main()
