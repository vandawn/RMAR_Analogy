#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_txt(path, colname):
    if path is None or not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep="\t", header=None, names=["ent_id", colname], dtype=str)
    # 去重：若同一 ent_id 多次出现，仅保留首次出现
    df = df.drop_duplicates(subset=["ent_id"], keep="first").reset_index(drop=True)
    return df

def encode_texts(texts, model, batch_size=128, max_seq_len=384, normalize=True):
    # SentenceTransformer 的 max_seq_length 控制截断长度
    model.max_seq_length = max_seq_len
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # L2 归一化
    )
    return emb  # np.ndarray, [N, D]

def main():
    parser = argparse.ArgumentParser(description="Generate (long and/or short) text embeddings for entities.")
    parser.add_argument("--long_file", type=str, default="entity2textlong.txt",
                        help="Path to entity2textlong.txt")
    parser.add_argument("--short_file", type=str, default=None,
                        help="Path to entity2text.txt (optional). If provided, will use both texts.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-mpnet-base-v2",
                        help="Sentence-Transformers model name.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--combine", type=str, default="auto",
                        choices=["auto", "long", "concat", "avg"],
                        help=(
                            "auto: 若提供了 short_file 则 concat，否则仅 long；"
                            "long: 只用长文本；"
                            "concat: 长短拼接（1536维，缺失短文本用0向量填充）；"
                            "avg: 长短平均（768维，缺失短文本则仅用长文本）"
                        ))
    parser.add_argument("--normalize_input", action="store_true",
                        help="对单个长/短文本嵌入做 L2 归一化（默认关，建议开）")
    parser.add_argument("--normalize_output", action="store_true",
                        help="对最终输出向量（拼接或平均后）再做一次 L2 归一化（默认关）")
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="可选：PCA 降维到该维度（0 表示不做 PCA）")
    parser.add_argument("--out_path", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/embeddings/MCNet-textual.pt",
                        help="Where to save the final tensor")
    parser.add_argument("--out_ids", type=str,
                        default="/data/rwan551/code/SNAG_Analogy/embeddings/MCNet-textual.ids",
                        help="Where to save the ent_id list (one per line)")

    args = parser.parse_args()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device: {device}")

    # 读入数据
    df_long = load_txt(args.long_file, "text_long")
    if df_long is None:
        raise FileNotFoundError(f"long_file not found: {args.long_file}")

    df_short = load_txt(args.short_file, "text_short") if args.short_file else None

    # 合并（outer 保留所有 ent_id；short 不在的用 NaN）
    if df_short is not None:
        df = pd.merge(df_long, df_short, on="ent_id", how="outer")
        print(f"[Info] merged entities: {len(df)} (long={len(df_long)}, short={len(df_short)})")
    else:
        df = df_long.copy()
        print(f"[Info] using long only: {len(df)} entities")

    # 决定组合策略
    if args.combine == "auto":
        combine = "concat" if df_short is not None else "long"
    else:
        combine = args.combine
    print(f"[Info] combine strategy: {combine}")

    # 加载模型
    print(f"[Info] loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)

    # --- 编码长文本 ---
    long_texts = df["text_long"].fillna("").astype(str).tolist()
    print("[Info] encoding LONG texts...")
    emb_long = encode_texts(
        long_texts, model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_length,
        normalize=args.normalize_input
    )  # [N, 768]
    assert emb_long.ndim == 2
    N, D = emb_long.shape

    # --- 编码短文本（按需） ---
    emb_short = None
    if combine in ("concat", "avg"):
        if "text_short" in df.columns:
            # 只对非空短文本编码；缺失短文本按策略处理
            mask_short = df["text_short"].notna() & (df["text_short"].astype(str).str.strip() != "")
            idx = np.where(mask_short.values)[0]
            emb_short = np.zeros((N, D), dtype=np.float32)  # 先全 0
            if len(idx) > 0:
                short_texts = df.loc[mask_short, "text_short"].astype(str).tolist()
                print(f"[Info] encoding SHORT texts for {len(idx)} / {N} entities...")
                emb_short_subset = encode_texts(
                    short_texts, model,
                    batch_size=args.batch_size,
                    max_seq_len=args.max_seq_length,
                    normalize=args.normalize_input
                )
                emb_short[idx] = emb_short_subset.astype(np.float32)
            else:
                print("[Warn] no valid short texts found; emb_short will be zeros.")
        else:
            # 没有短文本列
            emb_short = None

    # --- 组合 ---
    if combine == "long":
        final = emb_long.astype(np.float32)
    elif combine == "avg":
        if emb_short is None:
            # 无短文本，退化为只用长文本
            final = emb_long.astype(np.float32)
        else:
            # 对有短文本的样本，取平均；对缺失短文本（全 0）的，退化为长文本
            final = emb_long + emb_short
            # 对缺失短文本的位置，需要除以 1 而不是 2（避免幅度变小）
            denom = np.ones((N, 1), dtype=np.float32) * 2.0
            # 哪些是缺失短文（全0）？
            miss_short = (np.linalg.norm(emb_short, axis=1, keepdims=True) == 0.0)
            denom[miss_short] = 1.0
            final = (final / denom).astype(np.float32)
    elif combine == "concat":
        if emb_short is None:
            # 无短文本，右侧用 0 向量，保持 1536 维维度一致性
            emb_short = np.zeros((N, D), dtype=np.float32)
        final = np.concatenate([emb_long, emb_short], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown combine: {combine}")

    # --- 可选：对最终向量整体再做 L2 归一化 ---
    if args.normalize_output:
        norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-12
        final = (final / norms).astype(np.float32)

    # --- 可选：PCA 降维 ---
    if args.pca_dim and args.pca_dim > 0:
        try:
            from sklearn.decomposition import PCA
        except Exception as e:
            raise ImportError("PCA 需要 scikit-learn，请先安装：pip install scikit-learn") from e
        print(f"[Info] applying PCA to {args.pca_dim} dims...")
        pca = PCA(n_components=args.pca_dim, random_state=0)
        final = pca.fit_transform(final).astype(np.float32)

    # 保存
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(torch.from_numpy(final), args.out_path)
    df[["ent_id"]].to_csv(args.out_ids, index=False, header=False)

    print(f"[Done] saved embeddings to: {args.out_path}")
    print(f"[Done] saved ent_id list to: {args.out_ids}")
    print(f"[Shape] {final.shape} (dtype=float32)")
    if combine == "concat":
        print("[Note] concat -> 1536 维（如 mpnet 768*2）；缺失短文本的位置右半部分为全 0。")
    elif combine == "avg":
        print("[Note] avg -> 768 维；缺失短文本则等同只用长文本。")
    else:
        print("[Note] long-only -> 768 维。")

if __name__ == "__main__":
    main()
