# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import pickle
# import numpy as np

# PICKLE_PATHS = [
#     "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/analogy/train.pickle",
#     "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/analogy/valid.pickle",
#     "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/analogy/test.pickle",
# ]

# for p in PICKLE_PATHS:
#     print("\n" + "=" * 88)
#     print("FILE:", p)
#     with open(p, "rb") as f:
#         obj = pickle.load(f)

#     if isinstance(obj, np.ndarray):
#         print("shape:", obj.shape, "dtype:", obj.dtype)
#         k = min(5, obj.shape[0])
#         print("\n-- first {} rows (raw) --".format(k))
#         print(obj[:k])

#         # 如果是浮点数组，顺便给一份转 int 的视图，便于看作 ID
#         if np.issubdtype(obj.dtype, np.floating):
#             print("\n-- first {} rows (as int) --".format(k))
#             print(obj[:k].astype(np.int64))
#     else:
#         print("type:", type(obj))
#         # 尝试打印前 5 个元素/项
#         try:
#             if isinstance(obj, list):
#                 print(obj[:5])
#             elif isinstance(obj, dict):
#                 for i, (k, v) in enumerate(list(obj.items())[:5], 1):
#                     print(f"[{i}] {k!r} -> {type(v).__name__}")
#             else:
#                 print(repr(obj)[:400])
#         except Exception as e:
#             print("preview failed:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

ANALOGY_DIR = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/analogy"
PATH_TO_SKIP   = os.path.join(ANALOGY_DIR, "to_skip.pickle")
PATH_PROBAS    = os.path.join(ANALOGY_DIR, "probas.pickle")
PATH_1to1      = os.path.join(ANALOGY_DIR, "analogy_1_1_triples.pickle")
PATH_ENT_ID    = os.path.join(ANALOGY_DIR, "ent_id")
PATH_REL_ID    = os.path.join(ANALOGY_DIR, "rel_id")

def load_id_map(path):
    """读取 '<key>\t<id>' -> 两个映射: id2key, key2id（容错空行/坏行）"""
    if not os.path.exists(path):
        return {}, {}
    id2key, key2id = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or "\t" not in ln:  # 跳过空行/坏行
                continue
            key, sid = ln.split("\t", 1)
            try:
                i = int(sid)
            except ValueError:
                continue
            id2key[i] = key
            key2id[key] = i
    return id2key, key2id

def peek_to_skip():
    print("\n" + "="*88)
    print("to_skip.pickle:", PATH_TO_SKIP)
    sk = pickle.load(open(PATH_TO_SKIP, "rb"))
    lhs = sk.get("lhs", {})
    rhs = sk.get("rhs", {})
    print(f"- keys: lhs={len(lhs)}  rhs={len(rhs)}")

    # 读取映射做解码
    id2ent, _ = load_id_map(PATH_ENT_ID)
    id2rel, _ = load_id_map(PATH_REL_ID)
    n_rel = (max(id2rel.keys()) + 1) if id2rel else 0

    # 取前 3 条样例（lhs 侧）
    print("\n-- lhs side (key = (rhs_id, rel+n_rel)) --")
    for i, (k, v) in enumerate(list(lhs.items())[:3], 1):
        try:
            rhs_id, rel_plus = k
            rel_id = rel_plus - n_rel if n_rel else rel_plus
        except Exception:
            print(f"[{i}] key(raw)={k} -> value_len={len(v)} (unable to parse)")
            continue
        rhs_name = id2ent.get(rhs_id, f"<E{rhs_id}>")
        rel_name = id2rel.get(rel_id, f"<R{rel_id}>")
        sample_vals = v[:5]
        sample_names = [id2ent.get(x, f"<E{x}>") for x in sample_vals]
        print(f"[{i}] key: (rhs={rhs_id}:{rhs_name}, rel={rel_id}:{rel_name}) -> "
              f"value_len={len(v)}, value_sample={sample_vals} / {sample_names}")

    # 取前 3 条样例（rhs 侧）
    print("\n-- rhs side (key = (lhs_id, rel)) --")
    for i, (k, v) in enumerate(list(rhs.items())[:3], 1):
        try:
            lhs_id, rel_id = k
        except Exception:
            print(f"[{i}] key(raw)={k} -> value_len={len(v)} (unable to parse)")
            continue
        lhs_name = id2ent.get(lhs_id, f"<E{lhs_id}>")
        rel_name = id2rel.get(rel_id, f"<R{rel_id}>")
        sample_vals = v[:5]
        sample_names = [id2ent.get(x, f"<E{x}>") for x in sample_vals]
        print(f"[{i}] key: (lhs={lhs_id}:{lhs_name}, rel={rel_id}:{rel_name}) -> "
              f"value_len={len(v)}, value_sample={sample_vals} / {sample_names}")

def peek_probas():
    print("\n" + "="*88)
    print("probas.pickle:", PATH_PROBAS)
    probs = pickle.load(open(PATH_PROBAS, "rb"))
    arr_lhs = probs.get("lhs", None)
    arr_rhs = probs.get("rhs", None)
    arr_both = probs.get("both", None)
    if isinstance(arr_lhs, np.ndarray):
        print(f"- lhs:  shape={arr_lhs.shape}, dtype={arr_lhs.dtype}, sum={arr_lhs.sum():.6f}")
    if isinstance(arr_rhs, np.ndarray):
        print(f"- rhs:  shape={arr_rhs.shape}, dtype={arr_rhs.dtype}, sum={arr_rhs.sum():.6f}")
    if isinstance(arr_both, np.ndarray):
        print(f"- both: shape={arr_both.shape}, dtype={arr_both.dtype}, sum={arr_both.sum():.6f}")

    # 打印 both 概率最高的前 5 个实体
    if isinstance(arr_both, np.ndarray):
        id2ent, _ = load_id_map(PATH_ENT_ID)
        idx = np.argsort(-arr_both)[:5]
        print("\n-- top-5 by 'both' prob --")
        for rank, i in enumerate(idx, 1):
            name = id2ent.get(int(i), f"<E{i}>")
            print(f"[{rank}] id={int(i):6d}  prob={arr_both[i]:.6f}  name={name}")

def peek_1to1():
    print("\n" + "="*88)
    print("analogy_1_1_triples.pickle:", PATH_1to1)
    obj = pickle.load(open(PATH_1to1, "rb"))
    id2ent, _ = load_id_map(PATH_ENT_ID)
    id2rel, _ = load_id_map(PATH_REL_ID)

    if isinstance(obj, np.ndarray):
        print(f"- ndarray shape={obj.shape}, dtype={obj.dtype}")
        k = min(5, len(obj))
        print("\n-- first {} rows (IDs) --".format(k))
        print(obj[:k])
        if obj.ndim == 2 and obj.shape[1] >= 3:
            print("\n-- decoded --")
            for i in range(k):
                h, r, t = [int(x) for x in obj[i, :3]]
                hs = id2ent.get(h, f"<E{h}>")
                rs = id2rel.get(r, f"<R{r}>")
                ts = id2ent.get(t, f"<E{t}>")
                print(f"[{i+1}] ({hs}) --{rs}--> ({ts})")
    elif isinstance(obj, (list, tuple, set)):
        obj_list = list(obj)
        k = min(5, len(obj_list))
        print(f"- container type={type(obj).__name__}, size={len(obj_list)}")
        print("\n-- first {} items (raw) --".format(k))
        for i in range(k):
            print(f"[{i+1}] {obj_list[i]}")
        # 尝试解码三元组
        print("\n-- decoded (if triples) --")
        for i in range(k):
            tri = obj_list[i]
            try:
                h, r, t = int(tri[0]), int(tri[1]), int(tri[2])
                hs = id2ent.get(h, f"<E{h}>")
                rs = id2rel.get(r, f"<R{r}>")
                ts = id2ent.get(t, f"<E{t}>")
                print(f"[{i+1}] ({hs}) --{rs}--> ({ts})")
            except Exception:
                pass
    else:
        print(f"- type={type(obj)} (show repr head)")
        print(repr(obj)[:600])

if __name__ == "__main__":
    if os.path.exists(PATH_TO_SKIP):
        peek_to_skip()
    else:
        print("to_skip.pickle not found:", PATH_TO_SKIP)

    if os.path.exists(PATH_PROBAS):
        peek_probas()
    else:
        print("probas.pickle not found:", PATH_PROBAS)

    if os.path.exists(PATH_1to1):
        peek_1to1()
    else:
        print("analogy_1_1_triples.pickle not found:", PATH_1to1)

    print("\nDone.")
