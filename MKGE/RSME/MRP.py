# import pickle
# from pathlib import Path
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm


# def get_filtered_triples(base_path,output_file):
#     ent_1_1_triple=[]
#     root = Path(base_path)
#     files = ['wiki_tuple_ids.txt']

#     for f in files:
#         cnt_1, cnt_2, cnt_3, cnt_above_3 = 0, 0, 0, 0
#         triples = open(root / (f ), 'r').readlines()
#         triples=[triple.strip().split('\t') for triple in triples]
#         # triples = triples.tolist() # return type is np.array
#         num = len(triples)
#         heads = defaultdict(defaultdict)
#         for tri in triples:
#             h, r, t = tri[:]
#             h=h.strip()
#             r=r.strip()
#             t=t.strip()
#             if t not in heads[h]:
#                 heads[h][t] = []
#             heads[h][t].append((h,r,t))
#         for k, v in heads.items():
#             for kk, l in v.items():
#                 if len(l) == 1:
#                     cnt_1 += 1
#                     ent_1_1_triple.append(l[0])
#                 if len(l) == 2:
#                     cnt_2 += 2
#                 if len(l) == 3:
#                     cnt_3 += 3
#                 if len(l) > 3: cnt_above_3 += len(l)
#         print(f)
#         print('total triples: {}, 1-1-1: {}, 1-2-1: {}, 1-3-1: {}, 1-n-1:{}(n>3)'
#               .format(num, cnt_1, cnt_2, cnt_3, cnt_above_3))
#         cnt = [cnt_1, cnt_2, cnt_3, cnt_above_3]
#         ratio = [float(format(i / num, '.4f')) for i in cnt]
#         print('ratio: 1-1-1: {}, 1-2-1: {}, 1-3-1: {}, 1-n-1:{}(n>3)'.
#               format(ratio[0], ratio[1], ratio[2], ratio[3]))
#     out=open(output_file,'wb')
#     pickle.dump(ent_1_1_triple,out)
#     out.close()

# def get_rank(triples: list, img_vectors: dict, tails, filtered_tails):
#     triples.sort()
#     tails = sorted(list(tails)) # tails
#     cur_ranks = []

#     h_t = defaultdict(list)   #dict，head--->tail
#     heads = set()
#     for tri in triples:
#         if tri[2] in filtered_tails:
#             h_t[tri[0]].append(tri[2])
#         heads.add(tri[0])
#     heads = sorted(list(heads))
#     for h in heads:
#         if h not in img_vectors or h_t[h] == []:
#             continue
#         head_norm_vector = np.tile(np.array(img_vectors[h])/np.sqrt(np.sum(img_vectors[h] ** 2)),(len(filtered_tails), 1))
#         scores = np.sum(head_norm_vector * tail_vectors, axis=1)
#         true_tail_idx = [filtered_tails.index(t) for t in h_t[h]]
#         score_rank = np.argsort(-scores).tolist()
#         ranks = [1 + score_rank.index(i) for i in true_tail_idx]
#         cur_avg_rank = sum(ranks) / len(ranks)
#         cur_ranks.append(cur_avg_rank / len(score_rank))
#     if len(cur_ranks) == 0:
#         return 0, '0'
#     return sum(cur_ranks) / len(cur_ranks), '%.1f' % cur_avg_rank + '/' + str(len(tails))


# def calculate_MRP(img_vec_file='fb15k_vgg16.pickle',triples_file='fb15k_1_1_triples.pickle',output_file='fb15k_vgg_rank.txt',base_path='./'):
#     root = Path(base_path)
#     img_vec = pickle.load(open(root / img_vec_file , 'rb'))
#     triples_all = pickle.load(open(root / triples_file, 'rb'))
#     rel_triples = {}
#     all_ranks, rels, ratio = [], [], []

#     ent_vec = {k.split('/')[-2]: v for k, v in img_vec.items()} #change image address format
#     tail_ent = set() 
#     for triple in triples_all: 
#         triple = [i.strip(' ') for i in triple]
#         h, r, t = triple
#         r_list = rel_triples.get(r, list())
#         r_list.append(triple)
#         rel_triples[r] = r_list
#         tail_ent.add(t)
#     #
#     filtered_tails = []
#     for i, t in enumerate(tail_ent):
#         if t in ent_vec.keys():
#             filtered_tails.append(t) 
#     #
#     global tail_vectors
#     tail_vectors = []
#     for i, t in enumerate(tail_ent):
#         if t in ent_vec.keys():
#             tail_vectors.append(ent_vec[t].reshape(1, -1))
#     tail_vectors = np.concatenate(tail_vectors, axis=0)
    
#     cnt = 1
#     for rel, triples in tqdm(rel_triples.items()):
#         print('process relations: {}/{}'.format(cnt, len(rel_triples.keys())))
#         score, ratio_str = get_rank(triples, ent_vec, tail_ent, filtered_tails)  #tirples: all triples under this relation, ent_vec:dict, path-entity vector, tail_ent: all tail entities
#         print(score,ratio_str)
#         rels.append(rel)
#         all_ranks.append(score)
#         ratio.append(ratio_str)
#         cnt += 1
#     avg_rank = sum(all_ranks) / len(all_ranks)
#     with open(root / output_file, 'w') as f:
#         for i in range(len(all_ranks)):
#             f.write('relation id: ' + str(rels[i]) + '\t'
#                     + 'rank: ' + str(all_ranks[i]) + '\t'
#                     + 'percentage: ' + str(ratio[i]) + '\n')
#         f.write('average rank: ' + str(avg_rank))
#     print('average rank:{}'.format(avg_rank))


# if __name__ == '__main__':
#     get_filtered_triples('../MarT/MarKG', 'analogy_1_1_triples.pickle')
#     calculate_MRP(img_vec_file='analogy_vit_best_img_vec.pickle', 
#                   triples_file='analogy_1_1_triples.pickle',
#                   output_file='analogy_vit_rank.txt')


# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ===== 路径按你的环境配置好 =====
ROOT = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME"
DATA_DIR = os.path.join(ROOT, "data/MCNet")

# 结构三元组（字符串）：每行 "h \t r \t t"
TRIPLES_TXT = os.path.join(ROOT, "src_data/MCNet/wiki_tuple_ids")

# 选图映射：{ entity_str -> "<encoded_folder>/<file>" }（相对路径）
BEST_IMG = os.path.join(DATA_DIR, "analogy_best_img.pickle")
# 图片根目录
IMG_ROOT = "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images"
# 图片向量：{ abspath -> np.array(D,) }
IMG_VEC_DICT = os.path.join(DATA_DIR, "analogy_vit_best_img_vec.pickle")

# 输出
OUT_1TO1   = os.path.join(DATA_DIR, "analogy_1_1_triples.pickle")
OUT_RANK   = os.path.join(DATA_DIR, "analogy_vit_rank.txt")

SEED = 42
NEG_K = 100
EPS = 1e-8


# ---------------- 1) 生成 1–1 三元组（按关系） ----------------
def make_1to1_triples_from_txt(triples_txt: str, out_pickle: str):
    """
    在同一关系 r 下筛选 1–1：
      对 (r,h) 只对应 1 个 t，且 对 (r,t) 只来源 1 个 h。
    输出为 list[ (h_str, r_str, t_str) ]（与 analogy 的文件一致）。
    """
    triples = []
    with open(triples_txt, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))

    rh_count = defaultdict(int)  # (r,h) -> #tails
    rt_count = defaultdict(int)  # (r,t) -> #heads
    for h, r, t in triples:
        rh_count[(r, h)] += 1
        rt_count[(r, t)] += 1

    out = []
    for h, r, t in triples:
        if rh_count[(r, h)] == 1 and rt_count[(r, t)] == 1:
            out.append((h, r, t))

    with open(out_pickle, "wb") as f:
        pickle.dump(out, f)

    print(f"[1to1] read={len(triples)}  kept(1-1)={len(out)}  -> {out_pickle}")
    return out_pickle


# ---------------- 2) 计算每个关系的 MRP ----------------
def build_entity_vectors(best_img_pkl: str, path2vec_pkl: str, img_root: str):
    """
    用 best_img 映射把 entity_str 直接对上向量：
      ent2vec[entity_str] = path2vec[ join(img_root, relpath) ]
    没向量的实体返回 None。
    """
    ent2rel = pickle.load(open(best_img_pkl, "rb"))
    path2vec = pickle.load(open(path2vec_pkl, "rb"))

    ent2vec = {}
    dim = None
    miss = 0
    for ent, relp in ent2rel.items():
        absp = os.path.join(img_root, relp)
        vec = path2vec.get(absp)
        if vec is None:
            ent2vec[ent] = None
            miss += 1
        else:
            v = np.asarray(vec, dtype=np.float32)
            if dim is None:
                dim = v.size
            ent2vec[ent] = v
    print(f"[vectors] entities={len(ent2rel)}  hit={len(ent2rel)-miss}  miss={miss}")
    return ent2vec, (dim or 1000)


def cosine_sim(a: np.ndarray, B: np.ndarray):
    """
    a: (D,) 已归一化； B: (N,D) 已归一化
    返回 (N,) 的点积 = cos 相似度
    """
    return B @ a


def compute_mrp(triples_1to1_pkl: str,
                best_img_pkl: str,
                img_vec_dict_pkl: str,
                img_root: str,
                out_rank_txt: str,
                neg_k: int = NEG_K):
    """
    逻辑：
      - 先把 entity -> 向量 建好（单位向量）
      - 1-1 三元组按关系分组
      - 对每个关系 r：
          * 构建“尾实体池”（有向量的尾）
          * 对每个头（有向量、且至少一个有效尾），
            计算真尾在负样本（尾池）中的百分位名次，求平均
      - 输出： relation id: <r>  rank: <mrp>  percentage: <usable>/<total>
    """
    triples = pickle.load(open(triples_1to1_pkl, "rb"))  # list[(h,r,t)] (strings)
    ent2vec, dim = build_entity_vectors(best_img_pkl, img_vec_dict_pkl, img_root)

    # 单位化缓存
    norm_cache = {}
    def get_unit_vec(ent):
        if ent not in norm_cache:
            v = ent2vec.get(ent)
            if v is None:
                norm_cache[ent] = None
            else:
                n = np.linalg.norm(v) + EPS
                norm_cache[ent] = v / n
        return norm_cache[ent]

    # 按关系分组
    by_rel = defaultdict(list)
    for h, r, t in triples:
        by_rel[r].append((h, t))

    lines = []
    for rid, pairs in tqdm(by_rel.items(), desc="MRP per relation"):
        total = len(pairs)

        # 尾池：有向量的尾实体（去重）
        tails = sorted({t for _, t in pairs if get_unit_vec(t) is not None})
        if not tails:
            lines.append(f"relation id: {rid}\trank: 1.0\tpercentage: 0.0/{total}")
            continue
        tail_mat = np.vstack([get_unit_vec(t) for t in tails])  # (T, D)

        # 逐头计算
        ranks = []
        used = 0
        for h, t in pairs:
            vh = get_unit_vec(h)
            vt = get_unit_vec(t)
            if vh is None or vt is None:
                continue
            # 负样本：从 tails 采样，排除真尾
            # （若 tails 很小，直接全体）
            if len(tails) <= 1:
                continue
            cand_idx = np.arange(len(tails), dtype=np.int64)
            # 排除真尾
            true_idx = tails.index(t)
            cand_idx = cand_idx[cand_idx != true_idx]
            if cand_idx.size == 0:
                continue
            if cand_idx.size > neg_k:
                # 固定采样，保证不同头公平
                rng = np.random.default_rng(SEED)
                cand_idx = rng.choice(cand_idx, size=neg_k, replace=False)

            # 计算分数
            sims_neg = cosine_sim(vh, tail_mat[cand_idx])  # (K,)
            sim_true = float(np.dot(vh, vt))               # 标量
            rank = float((sims_neg >= sim_true).sum()) / float(len(sims_neg))
            ranks.append(rank)
            used += 1

        mrp = float(np.mean(ranks)) if ranks else 1.0
        lines.append(f"relation id: {rid}\trank: {mrp}\tpercentage: {float(used)}/{total}")

    with open(out_rank_txt, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"[OK] wrote {out_rank_txt} ({len(lines)} relations)")
    return out_rank_txt


if __name__ == "__main__":
    # 1) 生成 1–1 三元组（按关系）
    make_1to1_triples_from_txt(TRIPLES_TXT, OUT_1TO1)

    # 2) 计算 MRP 并写 rank.txt
    compute_mrp(
        triples_1to1_pkl=OUT_1TO1,
        best_img_pkl=BEST_IMG,
        img_vec_dict_pkl=IMG_VEC_DICT,
        img_root=IMG_ROOT,
        out_rank_txt=OUT_RANK
    )
