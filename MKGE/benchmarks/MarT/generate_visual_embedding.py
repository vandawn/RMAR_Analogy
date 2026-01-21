import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np

# 读取文件到DataFrame
df = pd.read_csv('entity2textlong.txt', sep='\t', header=None)
df.columns = ['ent_id', 'text']

img_emb_path = "/home/rwan551/Code/SNAG/analogy_embeddings/analogy-img"

print(df)

# 加载图像嵌入向量
img_emb = []
no_embed = 0
for index, row in df.iterrows():
    entity_id = row['ent_id']
    try:
        with open(f"{img_emb_path}/{entity_id}/avg_embedding.pkl", "rb") as f:
            emb = pickle.load(f)
            img_emb.append(torch.tensor(emb, dtype=torch.float32))  # 确保将嵌入转换为Tensor

    except:
        no_embed += 1
        random_emb = np.random.normal(0, 1, 4096)
        img_emb.append(torch.tensor(random_emb, dtype=torch.float32))  # 生成Tensor而不是numpy数组
        


print(f"Number of entities without embeddings: {no_embed}")

# 将 img_emb 列表转换为张量
img_emb = torch.stack(img_emb, dim=0)
print(img_emb)
print(img_emb.shape)

# 指定保存路径
pth_file_path = '/home/rwan551/Code/SNAG/embeddings/Analogy-visual.pth'

# 保存嵌入向量到 .pth 文件
torch.save(img_emb, pth_file_path)

print(f"Visual embeddings saved to {pth_file_path}")
