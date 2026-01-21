#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pickle
import unicodedata
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch


def load_entities_flex(path: str) -> pd.DataFrame:
    """
    读取 entity2text.txt（或同格式），兼容以 Tab 或若干空格分隔的两列：
      ent_id <whitespace> text
    只保留 ent_id 和 text 两列，ent_id 做 NFKC 规范化与去空白。
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            ent_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            ent_id = unicodedata.normalize("NFKC", ent_id.strip())
            rows.append((ent_id, text))
    df = pd.DataFrame(rows, columns=["ent_id", "text"]).drop_duplicates("ent_id")
    return df


def to_fixed_dim_tensor(x, dim: int, warn_prefix: str = "", max_warn: list = None) -> torch.Tensor:
    """
    将任意嵌入对象转为 1D float32 Tensor，并确保长度=dim。
    若长度不匹配：>dim 截断，<dim 右侧零填充（最多打印3次警告）。
    """
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == dim:
        return torch.from_numpy(arr)
    if max_warn is not None and max_warn[0] < 3:
        print(f"[Warn]{warn_prefix} embed dim={arr.size} != {dim}, "
              f"{'truncate' if arr.size > dim else 'pad zeros'} to {dim}.")
        max_warn[0] += 1
    if arr.size > dim:
        arr = arr[:dim]
    else:
        pad = np.zeros(dim - arr.size, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return torch.from_numpy(arr)


def main():
    parser = argparse.ArgumentParser(description="Load per-entity visual embeddings (URL-encoded folder names).")
    parser.add_argument("--entities_file", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/benchmarks/MCNet/entity2text.txt",
                        help="包含 ent_id 和 text 的两列文本文件。")
    parser.add_argument("--embeds_dir", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/benchmarks/MCNet/visual_embeds",
                        help="每个实体一个文件夹（文件夹名为 URL 编码后的 ent_id），里面有 avg_embedding.pkl")
    parser.add_argument("--embed_filename", type=str, default="avg_embedding.pkl",
                        help="每个实体目录下的嵌入文件名。")
    parser.add_argument("--dim", type=int, default=4096, help="视觉向量维度（缺失或不匹配时用于填充/截断）。")
    parser.add_argument("--out_path", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/embeddings/MCNet-visual.pt",
                        help="输出 .pt 路径（torch.save）。")
    parser.add_argument("--missing_ids_path", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/embeddings/MCNet-visual.missing_ids.txt",
                        help="缺失嵌入的 ent_id 清单。")
    args = parser.parse_args()

    # 1) 读取实体列表
    df = load_entities_flex(args.entities_file)
    print(f"[Info] entities: {len(df)}  file: {args.entities_file}")

    img_tensors = []
    missing_ids = []
    warn_counter = [0]  # 控制维度警告次数

    # 2) 遍历实体，按 URL 编码规则定位 pkl
    for i, row in df.iterrows():
        ent_id = row["ent_id"]
        enc = quote(ent_id, safe="")  # 重要：把 / 等都编码掉 → %2F
        pkl_path = os.path.join(args.embeds_dir, enc, args.embed_filename)

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)
                vec = to_fixed_dim_tensor(obj, args.dim, warn_prefix=f" {ent_id}", max_warn=warn_counter)
            except Exception as e:
                # 读到了但坏了：用随机向量顶上，并记录
                print(f"[Warn] failed to load {pkl_path}: {e}  → use random N(0,1)")
                vec = torch.randn(args.dim, dtype=torch.float32)
                missing_ids.append(ent_id + "  # corrupt?")
        else:
            # 完全不存在：随机向量
            vec = torch.randn(args.dim, dtype=torch.float32)
            missing_ids.append(ent_id)

        img_tensors.append(vec)

        if (i + 1) % 1000 == 0:
            print(f"[Prog] processed {i + 1}/{len(df)}")

    # 3) 堆叠与保存
    img_emb = torch.stack(img_tensors, dim=0).contiguous()  # [N, dim]
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(img_emb, args.out_path)
    print(f"[Done] Visual embeddings saved to {args.out_path}  |  shape={tuple(img_emb.shape)}  dtype={img_emb.dtype}")

    # 4) 缺失清单
    if missing_ids:
        os.makedirs(os.path.dirname(args.missing_ids_path), exist_ok=True)
        with open(args.missing_ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_ids))
        print(f"[Info] Number of entities without embeddings: {len(missing_ids)}")
        print(f"[Info] Missing list saved to {args.missing_ids_path}")
    else:
        print("[Info] All entities have visual embeddings.")


if __name__ == "__main__":
    main()
