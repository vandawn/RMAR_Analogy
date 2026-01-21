# import numpy as np
# import os
# import random
# import pickle
# import math


# def get_img_vec_array(proportion,img_vec_path='fb15k_vit.pickle',eutput_file='img_vec_id_fb15k_{}_vit.pickle',dim=1000):
#     img_vec=pickle.load(open(img_vec_path,'rb'))
#     img_vec={k.split('/')[-2]:v for k,v in img_vec.items()}
#     f=open('src_data/Analogy/wiki_tuple_ids', 'r')
#     Lines=f.readlines()
#     entities = set()
#     for line in Lines:
#         head, rel, tail = line.split('\t')
#         entities.add(head)
#         entities.add(tail.replace('\n', ''))

#     id2ent={}
#     img_vec_array=[]
#     for id, ent in enumerate(entities):
#         id2ent[id]=ent
#         if ent in img_vec.keys():
#             print(id, ent)
#             img_vec_array.append(img_vec[ent])
#         else:
#             img_vec_array.append([0 for i in range(dim)])
#     img_vec_by_id = np.array(img_vec_array)
#     out=open(eutput_file,'wb')
#     pickle.dump(img_vec_by_id,out)
#     out.close()


# def get_img_vec_array_forget(proportion,remember_proportion,rank_file='fb15k_vit_rank.txt',eutput_file='rel_MPR_PD_vit_{}_mrp{}.pickle'):
#     with open(rank_file,'r') as f:
#         Ranks=f.readlines()
#         rel_rank={}
#         for r in Ranks:
#             try:
#                 rel,mrp=r.strip().split('\t')
#             except Exception as e:
#                 print(e)
#                 print(r)
#                 continue
#             rel_rank[rel[10:]]=float(mrp[12:])

#     with open('../../MarT/dataset/MarKG/relation2text.txt', 'r') as f:
#         Lines=f.readlines()

#     rel_id_pd=[]
#     for l in Lines:
#         rel,_=l.strip().split('\t')
#         try:
#             if rel_rank[rel]<remember_proportion/100.0:
#                 rel_id_pd.append([1])
#             else:
#                 rel_id_pd.append([0])
#         except Exception as e:
#             print(e)
#             rel_id_pd.append([0])
#             continue

#     rel_id_pd=np.array(rel_id_pd)

#     with open(eutput_file.format(remember_proportion),'wb') as out:
#         pickle.dump(rel_id_pd,out)


# def get_img_vec_sig_alpha(proportion,rank_file='fb15k_vit_rank.txt',eutput_file='rel_MPR_SIG_vit_{}.pickle'):
#     with open(rank_file,'r') as f:
#         Ranks=f.readlines()
#         rel_rank={}
#         for r in Ranks:
#             try:
#                 rel,mrp=r.strip().split('\t')
#             except Exception as e:
#                 print(e)
#                 print(r)
#                 continue
#             rel_rank[rel[10:]]=float(mrp[12:])

#     with open('../../MarT/dataset/MarKG/relation2text.txt', 'r') as f:
#         Lines=f.readlines()

#     rel_sig_alpha=[]
#     for l in Lines:
#         rel,_=l.strip().split('\t')
#         try:
#             rel_sig_alpha.append([1/(1+math.exp(rel_rank[rel]))])
#         except Exception as e:
#             print(e)
#             rel_sig_alpha.append([1 / (1 + math.exp(1))])
#             continue

#     rel_id_pd=np.array(rel_sig_alpha)

#     with open(eutput_file,'wb') as out:
#         pickle.dump(rel_id_pd,out)

# def sample(proportion,data_path='./src_data/FB15K'):
#     with open(data_path+'/train') as f:
#         Ls=f.readlines()
#         L = [random.randint(0, len(Ls)-1) for _ in range(round(len(Ls)*proportion))]
#         Lf=[Ls[l] for l in L]

#     if not os.path.exists(data_path+'_{}/'.format(round(proportion*100))):
#         os.mkdir(data_path+'_{}/'.format(round(proportion*100)))
#     Ent=set()

#     with open(data_path+'_{}/train'.format(round(100*proportion)),'w') as f:
#         for l in Lf:
#             h,r,t=l.strip().split()
#             Ent.add(h)
#             Ent.add(r)
#             Ent.add(t)
#             f.write(l)
#             f.flush()

#     with open(data_path+'/valid','r') as f:
#         Ls = f.readlines()

#     with open(data_path+'_{}/valid'.format(round(100*proportion)),'w') as f:
#         for l in Ls:
#             h,r,t=l.strip().split()
#             if h in Ent and r in Ent and t in Ent:
#                 f.write(l)
#                 f.flush()
#             else:
#                 print(l.strip()+' pass')

#     with open(data_path+'/test','r') as f:
#         Ls = f.readlines()

#     with open(data_path+'_{}/test'.format(round(proportion*100)),'w') as f:
#         for l in Ls:
#             h, r, t = l.strip().split()
#             if h in Ent and r in Ent and t in Ent:
#                 f.write(l)
#                 f.flush()
#             else:
#                 print(l.strip()+' pass')
                
# def split_mkg_data(root_path):
#     for f in ['train', 'valid', 'test']:
#         path = open(f'{root_path}/{f}.pickle', 'rb')
#         data = pickle.load(path)
#         data = np.append(data, np.zeros((len(data), 1)), axis=-1)
#         for i in range(len(data)):
#             rnd = random.random()
#             if rnd <= 0.4:
#                 data[i][-1] = 0
#             elif rnd > 0.4 and rnd < 0.7:
#                 data[i][-1] = 1
#             else:
#                 data[i][-1] = 2
#         with open(f'{root_path}/{f}.pickle', 'wb') as f:
#             pickle.dump(data, f)

# if __name__ == '__main__':
#     get_img_vec_array(0, img_vec_path='data/analogy/analogy_vit_best_img_vec.pickle', eutput_file='data/analogy/img_vec_id_analogy_vit.pickle')
#     get_img_vec_sig_alpha(20, 'data/analogy/analogy_vit_rank.txt', 'data/analogy/rel_MPR_SIG_vit.pickle')
#     get_img_vec_array_forget(30, 100, 'data/analogy/analogy_vit_rank.txt', 'data/analogy/rel_MPR_PD_vit_mrp{}.pickle')
#     split_mkg_data('data/analogy')



# -*- coding: utf-8 -*-
import os
import re
import pickle
import math
import numpy as np

# ========= 路径（已按 MCNet 配好，可按需微调）=========
DATA_DIR = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet"

ENT_ID_PATH = os.path.join(DATA_DIR, "ent_id")
REL_ID_PATH = os.path.join(DATA_DIR, "rel_id")

# 实体 -> 相对图片路径
BEST_IMG_PICKLE = os.path.join(DATA_DIR, "analogy_best_img.pickle")
# 图片根目录
BASE_IMAGE_ROOT = "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images"
# 图片向量：{绝对路径: np.array(D,)}
IMG_VEC_DICT = os.path.join(DATA_DIR, "analogy_vit_best_img_vec.pickle")

# 关系排名文件（已生成）
RANK_FILE = os.path.join(DATA_DIR, "analogy_vit_rank.txt")

# 输出
OUT_IMG_VEC_ID = os.path.join(DATA_DIR, "img_vec_id_vit.pickle")
OUT_REL_SIG = os.path.join(DATA_DIR, "rel_MPR_SIG_vit.pickle")
OUT_REL_PD_FMT = os.path.join(DATA_DIR, "rel_MPR_PD_vit_mrp{}.pickle")
# ============================================


# ---------- 小工具 ----------
def _read_ent_id(path):
    ent2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            k, v = line.split("\t", 1)
            ent2id[k] = int(v)
    id_sorted = [None] * len(ent2id)
    for k, i in ent2id.items():
        id_sorted[i] = k
    return id_sorted, ent2id

def _read_rel_id(path):
    rel2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            k, v = line.split("\t", 1)
            rel2id[k] = int(v)
    id_sorted = [None] * len(rel2id)
    for k, i in rel2id.items():
        id_sorted[i] = k
    return id_sorted, rel2id

def _load_rank_file(path):
    """解析 'analogy_vit_rank.txt' 为 {relation_str: float_rank}，兼容多种格式。"""
    rel_rank = {}
    if not os.path.exists(path):
        print(f"[WARN] rank_file not found: {path} -> using defaults")
        return rel_rank
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s:
                continue
            # 优先匹配 "relation id: XXX   rank: YYY"
            m = re.search(r"relation id:\s*([^\t]+)\s+", s)
            m2 = re.search(r"rank:\s*([+-]?\d+(\.\d+)?)", s)
            if m and m2:
                rel = m.group(1).strip()
                val = float(m2.group(1))
                rel_rank[rel] = val
                continue
            # 兜底：尝试 "\t" 或空格分割的 "rel val"
            parts = re.split(r"[\t ]+", s)
            if len(parts) >= 2:
                try:
                    rel_rank[parts[0]] = float(parts[1])
                except Exception:
                    pass
            else:
                print(f"[WARN] cannot parse rank line: {s}")
    return rel_rank
# ---------------------------------------------


def get_img_vec_array_mcnet(img_vec_path=IMG_VEC_DICT,
                            best_img_pickle=BEST_IMG_PICKLE,
                            base_image_root=BASE_IMAGE_ROOT,
                            ent_id_path=ENT_ID_PATH,
                            output_file=OUT_IMG_VEC_ID,
                            dim=None):
    """
    生成按实体ID顺序对齐的图片向量矩阵：
      - 读取 {abs_img_path: np.array(D,)} 的向量字典
      - 读取 {entity: rel_path} 的 best_img 映射，用 base_image_root 拼出 abs_img_path
      - 对于 ent_id 中每个实体，若有向量则放入，否则用 0 向量填充
      - 存为 (N, D) 的 numpy 数组 pickle
    """
    print("[step] build entity-aligned image vector matrix")

    if not os.path.exists(img_vec_path):
        raise FileNotFoundError(f"img_vec dict not found: {img_vec_path}")
    if not os.path.exists(best_img_pickle):
        raise FileNotFoundError(f"best_img mapping not found: {best_img_pickle}")

    path2vec = pickle.load(open(img_vec_path, "rb"))
    ent2rel = pickle.load(open(best_img_pickle, "rb"))
    id_sorted_ents, ent2id = _read_ent_id(ent_id_path)

    # 推断维度
    if dim is None:
        try:
            dim = len(next(iter(path2vec.values())))
        except Exception:
            dim = 1000  # fallback

    out_mat = np.zeros((len(id_sorted_ents), dim), dtype=np.float32)

    miss, hit = 0, 0
    for ent in id_sorted_ents:
        idx = ent2id[ent]
        rel = ent2rel.get(ent)
        if not rel:
            miss += 1
            continue
        abs_path = os.path.join(base_image_root, rel)
        vec = path2vec.get(abs_path)
        if vec is None:
            miss += 1
            continue
        out_mat[idx] = np.asarray(vec, dtype=np.float32)
        hit += 1

    with open(output_file, "wb") as f:
        pickle.dump(out_mat, f)

    print(f"[img_vec_id] total_ents={len(id_sorted_ents)}  hit={hit}  miss={miss}  -> {output_file}")
    return output_file


def get_rel_sig_alpha_mcnet(rank_file=RANK_FILE,
                            rel_id_path=REL_ID_PATH,
                            output_file=OUT_REL_SIG):
    """
    生成每关系一个 sigmoid 权重：
      alpha_r = 1 / (1 + exp(rank_r))
    若 rank_file 不存在或缺该关系，默认 rank=1.0。
    输出形状: (R, 1)
    """
    print("[step] build relation SIG (sigmoid) weights")
    rels, rel2id = _read_rel_id(rel_id_path)
    rel_rank = _load_rank_file(rank_file)

    vals = []
    for rel in rels:
        r = rel_rank.get(rel, 1.0)
        vals.append([1.0 / (1.0 + math.exp(r))])
    arr = np.array(vals, dtype=np.float32)

    with open(output_file, "wb") as f:
        pickle.dump(arr, f)
    print(f"[rel_sig] R={len(rels)} -> {output_file}")
    return output_file


def get_rel_pd_mcnet(remember_proportion=100,
                     rank_file=RANK_FILE,
                     rel_id_path=REL_ID_PATH,
                     output_file_fmt=OUT_REL_PD_FMT):
    """
    生成每关系的“记住/忘记”0/1 标签：
      label_r = 1 if rank_r < (remember_proportion / 100.0) else 0
    若 rank_file 缺失或该关系无值，则默认 label=0。
    输出形状: (R, 1)
    """
    print(f"[step] build relation PD (threshold={remember_proportion}%) labels")
    rels, rel2id = _read_rel_id(rel_id_path)
    rel_rank = _load_rank_file(rank_file)

    thresh = remember_proportion / 100.0
    vals = []
    for rel in rels:
        r = rel_rank.get(rel, None)
        if r is None:
            vals.append([0])
        else:
            vals.append([1 if r < thresh else 0])
    arr = np.array(vals, dtype=np.int64)

    out_path = output_file_fmt.format(remember_proportion)
    with open(out_path, "wb") as f:
        pickle.dump(arr, f)
    print(f"[rel_pd] R={len(rels)} -> {out_path}")
    return out_path


# 幂等：不再默认追加随机列，保持 train/valid/test 原始三列；需要时可单独启用
def split_mkg_data_mcnet(root_path=DATA_DIR):
    print("[info] split_mkg_data_mcnet is disabled by default; enable manually if needed.")
    # 留空/占位，避免误操作


if __name__ == "__main__":
    # 1) 实体对齐的图片向量矩阵
    get_img_vec_array_mcnet()

    # 2) 关系先验（根据你的 rank 文件）
    get_rel_sig_alpha_mcnet(rank_file=RANK_FILE)
    get_rel_pd_mcnet(remember_proportion=100, rank_file=RANK_FILE)

    # 3) （可选）结构三元组追加随机 mode 列（默认不做）
    split_mkg_data_mcnet()
