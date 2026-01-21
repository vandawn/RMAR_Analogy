# # -*- coding: UTF-8 -*-
# '''
# Image encoder.  Get visual embeddings of images.
# '''
# import os
# import imagehash
# from PIL import Image
# import pickle
# from tqdm import tqdm
# class FilterGate():
#     def __init__(self,base_path,hash_size):
#         self.base_path=base_path
#         self.hash_size=hash_size
#         self.best_imgs={}

#     def phash_sim(self,img1,img2,hash_size=None):
#         if not hash_size:
#             hash_size=self.hash_size
#         img1_hash = imagehash.phash(Image.open(img1), hash_size=hash_size)
#         img2_hash = imagehash.phash(Image.open(img2), hash_size=hash_size)

#         return 1 - (img1_hash - img2_hash) / len(img1_hash) ** 2

#     def filter(self):
#         self.best_imgs={}
#         ents = os.listdir(self.base_path)
#         pbar = tqdm(total=len(ents))
#         while len(ents)>0:
#             ent=ents.pop()
#             imgs=os.listdir(self.base_path + ent + '/')
#             n_img=len(imgs)
#             if n_img == 0:
#                 pbar.update(1)
#                 continue
#             sim_matrix=[[0]*n_img for i in range(n_img)]
#             for i in range(n_img):
#                 for j in range(i+1,n_img):
#                     sim=self.phash_sim(self.base_path + ent + '/'+imgs[i], self.base_path + ent + '/'+imgs[j])
#                     sim_matrix[i][j]=sim
#                     sim_matrix[j][i] =sim
#             max_index=0
#             max_sim=sum(sim_matrix[0])
#             for i in range(1,n_img):
#                 if sum(sim_matrix[i])>max_sim:
#                     max_index=i
#                     max_sim=sum(sim_matrix[i])
#             self.best_imgs[ent]=self.base_path + ent + '/'+imgs[max_index]
#             pbar.update(1)
#         pbar.close()
#         return self.best_imgs

#     def save_best_imgs(self,output_file,n=1):
#         with open(output_file, 'wb') as out:
#             pickle.dump(self.best_imgs, out)




# if __name__ == '__main__':
#     f=FilterGate('../MarT/dataset/MARS/images/', hash_size=16)
#     f.filter()
#     f.save_best_imgs('analogy_best_img.pickle')








# -*- coding: UTF-8 -*-
"""
Select the best/central image per entity using pHash similarity (medoid).
Outputs a dict: {entity_str (unencoded): "<encoded_folder>/<filename>"}.
"""
import os
import pickle
from tqdm import tqdm
from PIL import Image
import imagehash
import urllib.parse

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class FilterGate:
    def __init__(self, base_path, hash_size=16):
        """
        base_path: 图片根目录，例如
          /home/.../MarT/dataset/MCNetAnalogy/images
        目录结构为：
          images/
            %2Fc%2Fen%2Foperation/
              1.jpg ... 5.jpg
        """
        self.base_path = base_path.rstrip("/")

        self.hash_size = hash_size
        self.best_imgs = {}  # {entity_str: "<encoded_folder>/<file>"}

    def _phash(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img, hash_size=self.hash_size)

    def _sim(self, p1, p2):
        """1 - HammingDistance / hash_len^2   (越大越相似)"""
        h1 = self._phash(p1)
        h2 = self._phash(p2)
        return 1 - (h1 - h2) / (len(h1) ** 2)

    def filter(self):
        """
        为每个实体（=每个编码后的子文件夹）挑一个“中心图”：
        使其与同文件夹内其它图片的相似度总和最大。
        """
        self.best_imgs = {}
        ents = [d for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d))]
        pbar = tqdm(total=len(ents))
        for enc_ent in ents:
            ent_dir = os.path.join(self.base_path, enc_ent)
            files = [f for f in os.listdir(ent_dir)
                     if os.path.splitext(f)[1].lower() in VALID_EXTS]
            if not files:
                pbar.update(1)
                continue
            files.sort()

            n = len(files)
            if n == 1:
                best_idx = 0
            else:
                # 计算“中心图”（naive：O(n^2)）
                simsums = [0.0] * n
                for i in range(n):
                    pi = os.path.join(ent_dir, files[i])
                    for j in range(i + 1, n):
                        pj = os.path.join(ent_dir, files[j])
                        try:
                            s = self._sim(pi, pj)
                        except Exception as e:
                            # 某张图坏了就当相似度 0
                            s = 0.0
                        simsums[i] += s
                        simsums[j] += s
                best_idx = max(range(n), key=lambda k: simsums[k])

            # key：解码后的实体字符串；value：相对路径（编码文件夹/文件名）
            ent_str = urllib.parse.unquote(enc_ent)               # "/c/en/operation"
            rel_path = os.path.join(enc_ent, files[best_idx])     # "%2Fc%2Fen%2Foperation/5.jpg"
            self.best_imgs[ent_str] = rel_path
            pbar.update(1)

        pbar.close()
        return self.best_imgs

    def save_best_imgs(self, output_file):
        with open(output_file, "wb") as out:
            pickle.dump(self.best_imgs, out)

if __name__ == "__main__":
    # === 按你的路径改好 ===
    BASE_PATH = "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images"
    OUT_PKL   = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet/analogy_best_img.pickle"

    fg = FilterGate(BASE_PATH, hash_size=16)
    fg.filter()
    fg.save_best_imgs(OUT_PKL)
    print(f"Saved {len(fg.best_imgs)} entries to: {OUT_PKL}")



