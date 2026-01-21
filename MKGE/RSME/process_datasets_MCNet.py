# -*- coding: utf-8 -*-
import os
import json
import errno
from pathlib import Path
import pickle
import numpy as np
import random
from collections import Counter

# ========= 路径配置（按你的机器已对齐）=========
DATA_ROOT = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data"

# finetune 用到的 ent/rel 映射
ENT_ID_PATH = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet/ent_id"
REL_ID_PATH = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet/rel_id"

# 你提供的 MCNetAnalogy jsonl / json 文件
JSON_FILES = {
    "train": "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/train.json",
    "dev":   "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/dev.json",
    "test":  "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/test.json",
}

# 输出目录
FT_OUT_DIR = os.path.join(DATA_ROOT, "MCNet")
# ============================================


def _load_ent_rel():
    """读取 ent_id / rel_id，返回 (ent2id, rel2id, id2ent, id2rel)"""
    with open(ENT_ID_PATH, "r", encoding="utf-8") as f:
        ent2id = {}
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, v = line.split("\t")
            ent2id[k] = int(v)

    with open(REL_ID_PATH, "r", encoding="utf-8") as f:
        rel2id = {}
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, v = line.split("\t")
            rel2id[k] = int(v)

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    print(f"[load] ent2id={len(ent2id)}, rel2id={len(rel2id)}")
    return ent2id, rel2id, id2ent, id2rel


def _load_jsonl_or_json(path):
    """支持两种格式：
       1) jsonl：每行一个 JSON 对象
       2) json：整个文件是一个 list
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().lstrip()
        if not txt:
            return []
        if txt[0] == "[":
            # 整个是一个 JSON 数组
            return json.loads(txt)
        # 否则按 jsonl 逐行解析
        f.seek(0)
        return [json.loads(line) for line in f if line.strip()]


def _print_split_stats(split, arr, id2ent, id2rel, input_count, mapped_count, skipped_count, sample_k=3):
    """打印数据统计与样例。arr 形状为 (N,6)。"""
    print("=" * 80)
    print(f"[{split}] 统计信息")
    print(f"- 输入样本数 (raw): {input_count}")
    print(f"- 成功映射数     : {mapped_count}")
    print(f"- 跳过数         : {skipped_count}")

    if arr.size == 0:
        print("- 数据为空，跳过更多统计与样例。")
        print("=" * 80)
        return

    # 唯一实体数（4列 union）
    ents = np.unique(np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]]))
    rels = np.unique(arr[:, 4])
    print(f"- 唯一实体数     : {len(ents)}")
    print(f"- 唯一关系数     : {len(rels)}")

    # mode 分布
    modes = arr[:, 5].astype(int).tolist()
    mode_counter = Counter(modes)
    mode_str = ", ".join([f"mode={m}: {c}" for m, c in sorted(mode_counter.items())])
    print(f"- mode 分布      : {mode_str}")

    # 样例
    print(f"\n[{split}] 样例展示（最多 {sample_k} 条）")
    idxs = list(range(arr.shape[0]))
    random.shuffle(idxs)
    idxs = idxs[: min(sample_k, len(idxs))]
    for i, idx in enumerate(idxs, 1):
        eh, et, q, a, r, m = arr[idx].tolist()
        eh_s = id2ent.get(eh, f"<UNK:{eh}>")
        et_s = id2ent.get(et, f"<UNK:{et}>")
        q_s  = id2ent.get(q,  f"<UNK:{q}>")
        a_s  = id2ent.get(a,  f"<UNK:{a}>")
        r_s  = id2rel.get(r,  f"<UNK:{r}>")

        print("-" * 80)
        print(f"样例 {i}:")
        print(f"ID版    : [eh={eh}, et={et}, q={q}, a={a}, rel={r}, mode={m}]")
        print(f"可读版  : example=({eh_s} , {et_s}) | question={q_s} | answer={a_s} | relation={r_s} | mode={m}")
    print("=" * 80)


def prepare_finetune_dataset():
    os.makedirs(FT_OUT_DIR, exist_ok=True)
    ent2id, rel2id, id2ent, id2rel = _load_ent_rel()

    for split, json_path in JSON_FILES.items():
        if not os.path.exists(json_path):
            print(f"[WARN] missing {split} file: {json_path} (skip)")
            continue

        data = _load_jsonl_or_json(json_path)
        mapped = []
        skipped = 0

        for obj in data:
            # 期望字段：example=[eh, et], question, answer, relation, mode
            try:
                eh, et = obj["example"][0], obj["example"][1]
                q, a = obj["question"], obj["answer"]
                rel, mode = obj["relation"], obj["mode"]
            except Exception:
                skipped += 1
                continue

            if (eh in ent2id and et in ent2id and
                q in ent2id and a in ent2id and rel in rel2id):
                mapped.append([
                    ent2id[eh],
                    ent2id[et],
                    ent2id[q],
                    ent2id[a],
                    rel2id[rel],
                    int(mode),
                ])
            else:
                skipped += 1

        arr = np.array(mapped, dtype="uint64")
        out_path = os.path.join(FT_OUT_DIR, f"{split}_ft.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(arr, f)

        print(f"[{split}] input={len(data)} mapped={len(mapped)} skipped={skipped} -> {out_path}")

        # —— 新增：统计信息与样例展示 ——
        _print_split_stats(
            split=split,
            arr=arr,
            id2ent=id2ent,
            id2rel=id2rel,
            input_count=len(data),
            mapped_count=len(mapped),
            skipped_count=skipped,
            sample_k=3
        )

    print(f"[done] outputs in: {FT_OUT_DIR}")


if __name__ == "__main__":
    try:
        prepare_finetune_dataset()
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print("File exists. skipping...")
        else:
            raise
