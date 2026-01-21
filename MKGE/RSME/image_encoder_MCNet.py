# -*- coding: UTF-8 -*-
"""
Image encoder. Get visual embeddings of images.
- 读取 analogy_best_img.pickle (实体 -> 相对图片路径)
- 用预训练视觉模型前向，得到每张图的向量（默认：ImageNet-1k logits, 1000维）
- 保存为 dict: {绝对图片路径: numpy向量} 的 pickle
"""
import os
import pickle
from tqdm import tqdm

import torch
import torch.cuda
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 可选：第三方 ViT（分类模型，输出1000维）
from pytorch_pretrained_vit import ViT

# ====== 路径配置（按你的环境已对齐）======
BEST_IMG_PICKLE = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet/analogy_best_img.pickle"
BASE_IMAGE_ROOT = "/home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images"
OUT_VEC_PICKLE  = "/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/data/MCNet/analogy_vit_best_img_vec.pickle"
# ========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class ImageEncoder:
    TARGET_IMG_SIZE = 384
    img_to_tensor = transforms.ToTensor()
    # 修复：3通道归一化（原来是 (0.5,), (0.5,)）
    Normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @staticmethod
    def get_embedding(self):
        pass

    def extract_feature(self, base_path, mapping_pickle):
        """
        base_path: 图片根目录
        mapping_pickle: analogy_best_img.pickle（值是相对路径）
        返回: dict {绝对图片路径: numpy向量}
        """
        self.model.eval()
        out_dict = {}

        with open(mapping_pickle, "rb") as f:
            best_img_map = pickle.load(f)

        # 只用 values()（相对路径），按你的原逻辑
        img_paths = [os.path.join(base_path, rel) for rel in list(best_img_map.values())]
        img_paths = [p for p in img_paths if os.path.exists(p)]
        print(f"Total images to encode: {len(img_paths)}")

        pbar = tqdm(total=len(img_paths))
        while len(img_paths) > 0:
            batch_files = []
            ok_files = []
            # 一次取至多5张，保持与你原逻辑一致（可根据显存改大）
            for _ in range(5):
                if not img_paths:
                    break
                fp = img_paths.pop()
                batch_files.append(fp)

            tensors = []
            for imgpath in batch_files:
                try:
                    # 修复：统一转 RGB，避免 1/4 通道问题
                    img = Image.open(imgpath).convert("RGB").resize((self.TARGET_IMG_SIZE, self.TARGET_IMG_SIZE))
                    img_tensor = self.img_to_tensor(img)
                    img_tensor = self.Normalizer(img_tensor)
                    # 现在一定是3通道；仍保留一次判断
                    if img_tensor.size(0) == 3:
                        tensors.append(img_tensor)
                        ok_files.append(imgpath)
                except Exception as e:
                    print(f"[WARN] read image failed: {imgpath} ({e})")
                    continue

            if len(tensors) == 0:
                pbar.update(len(batch_files))
                continue

            batch = torch.stack(tensors, 0).cuda(non_blocking=True)

            with torch.no_grad():  # 修复：不构图，省显存更快
                result = self.model(batch)

            # 兼容 torchvision 模型输出是 Tensor，第三方 ViT 也是 Tensor
            result_npy = result.detach().cpu().numpy()

            for i in range(len(result_npy)):
                out_dict[ok_files[i]] = result_npy[i]

            # 修复：按实际处理的数量更新
            pbar.update(len(batch_files))

        pbar.close()
        return out_dict


class VisionTransformer(ImageEncoder):
    def __init__(self):
        super().__init__()
        # 预训练 ViT-B/16 (ImageNet1k 分类头)
        self.model = ViT('B_16_imagenet1k', pretrained=True).cuda().eval()

    def get_embedding(self, base_path, mapping_pickle):
        self.d = self.extract_feature(base_path, mapping_pickle)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


class Resnet50(ImageEncoder):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True).cuda().eval()

    def get_embedding(self, base_path, mapping_pickle):
        self.d = self.extract_feature(base_path, mapping_pickle)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


class VGG16(ImageEncoder):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()

    def get_embedding(self, base_path, mapping_pickle):
        self.d = self.extract_feature(base_path, mapping_pickle)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


if __name__ == "__main__":
    # 选一个模型（ViT / ResNet50 / VGG16）
    model = VisionTransformer()
    # model = Resnet50()
    # model = VGG16()

    emb = model.get_embedding(BASE_IMAGE_ROOT, BEST_IMG_PICKLE)
    model.save_embedding(OUT_VEC_PICKLE)
    print(f"[OK] Saved {len(emb)} vectors to: {OUT_VEC_PICKLE}")
