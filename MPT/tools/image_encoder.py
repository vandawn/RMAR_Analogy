# -*- coding: utf-8 -*-
"""
Build analogy_entity2vec.pickle with ROI-style features: {entity_id: (36, D)}
- Images under URL-encoded subfolders: images_root/<urlencoded(entity_id)>/*.jpg
- Backends: resnet50(2048-D), clip(512-D), vilt(768-D)
- For each entity: average features across up to K images, then tile to (36, D)

Usage example (recommended resnet50 -> D=2048):
python tools/build_analogy_entity2vec_roi36.py \
  --images_root /home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/images \
  --entities_file /home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetKG/entity2text.txt \
  --out_pickle  /home/rwan551/code/MKG_Analogy-main/MarT/dataset/MCNetAnalogy/analogy_entity2vec.pickle \
  --backend resnet50 --max_per_entity 5 --batch_size 32 --fill_missing

If you pick clip/vilt, D != 2048; ensure your model config matches or add a projection layer.
"""
import os
import argparse
import urllib.parse
import pickle
from pathlib import Path
from typing import List, Dict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

# -------------------- I/O helpers --------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_entities(entity_file: Path) -> List[str]:
    ents = []
    with entity_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 第一列为 entity_id，后面列是描述文本
            eid = line.split('\t')[0]
            ents.append(eid)
    return ents

def list_entity_images(images_root: Path, entity_id: str, max_per_entity: int) -> List[Path]:
    folder = images_root / urllib.parse.quote(entity_id, safe="")
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort()  # 稳定顺序
    if max_per_entity and len(files) > max_per_entity:
        files = files[:max_per_entity]
    return files

# -------------------- Backends --------------------
class ResNet50Backend:
    """ResNet50 global pooled features (2048-D)."""
    def __init__(self, device, normalize=True):
        self.device = device
        self.normalize = normalize
        self.dim = 2048
        self.model = tv_models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, images: List[Image.Image], batch_size: int) -> np.ndarray:
        feats = []
        batch = []
        for img in images:
            try:
                batch.append(self.preprocess(img))
            except Exception:
                continue
            if len(batch) == batch_size:
                x = torch.stack(batch, 0).to(self.device)
                f = self.model(x).float().cpu().numpy()
                if self.normalize:
                    f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
                feats.append(f)
                batch = []
        if batch:
            x = torch.stack(batch, 0).to(self.device)
            f = self.model(x).float().cpu().numpy()
            if self.normalize:
                f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
            feats.append(f)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)

class CLIPBackend:
    """CLIP ViT-B/32 image features (512-D)."""
    def __init__(self, device, normalize=True, model_name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.normalize = normalize
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim  # 512

    @torch.no_grad()
    def embed(self, images: List[Image.Image], batch_size: int) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inputs).float().cpu().numpy()
            if self.normalize:
                f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
            feats.append(f)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)

class ViLTBackend:
    """ViLT-B/32 pooled features (768-D)."""
    def __init__(self, device, normalize=True, model_name="dandelin/vilt-b32-mlm"):
        from transformers import ViltModel, ViltProcessor
        self.device = device
        self.normalize = normalize
        self.model = ViltModel.from_pretrained(model_name).eval().to(device)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def embed(self, images: List[Image.Image], batch_size: int) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            texts = [""] * len(batch)  # ViLT 需要 text，占位空串
            inputs = self.processor(images=batch, text=texts, return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inputs)
            f = out.pooler_output.float().cpu().numpy()  # [B, D]
            if self.normalize:
                f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
            feats.append(f)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, type=str, help="图片根目录（包含 URL 编码子文件夹）")
    ap.add_argument("--entities_file", required=True, type=str, help="两列文本；第一列为 entity_id")
    ap.add_argument("--out_pickle", required=True, type=str, help="输出 pickle 路径（analogy_entity2vec.pickle）")
    ap.add_argument("--backend", choices=["resnet50", "clip", "vilt"], default="resnet50")
    ap.add_argument("--max_per_entity", type=int, default=5, help="每个实体最多取几张图做均值")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--roi_regions", type=int, default=36, help="ROI 数量，默认 36")
    ap.add_argument("--fill_missing", action="store_true", help="无图实体是否用 0 向量填充")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_l2norm", action="store_true", help="关闭 L2 归一化")
    args = ap.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    images_root = Path(args.images_root)
    entities_file = Path(args.entities_file)
    out_pickle = Path(args.out_pickle)
    out_pickle.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = not args.no_l2norm

    if args.backend == "resnet50":
        backend = ResNet50Backend(device=device, normalize=normalize)
    elif args.backend == "clip":
        backend = CLIPBackend(device=device, normalize=normalize)
    else:
        backend = ViLTBackend(device=device, normalize=normalize)

    entities = read_entities(entities_file)
    print(f"[Info] Entities: {len(entities)} | Backend: {args.backend} (D={backend.dim}) | ROI={args.roi_regions}")

    ent2vec: Dict[str, np.ndarray] = {}
    missing, ok = 0, 0

    for eid in tqdm(entities, desc="Extracting"):
        paths = list_entity_images(images_root, eid, args.max_per_entity)
        if not paths:
            missing += 1
            if args.fill_missing:
                ent2vec[eid] = np.zeros((args.roi_regions, backend.dim), dtype=np.float32)
            continue

        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                # 坏图跳过
                pass

        if not images:
            missing += 1
            if args.fill_missing:
                ent2vec[eid] = np.zeros((args.roi_regions, backend.dim), dtype=np.float32)
            continue

        feats = backend.embed(images, args.batch_size)   # [n, D]
        if feats.shape[0] == 0:
            missing += 1
            if args.fill_missing:
                ent2vec[eid] = np.zeros((args.roi_regions, backend.dim), dtype=np.float32)
            continue

        f_mean = feats.mean(axis=0).astype("float32")    # (D,)
        roi = np.repeat(f_mean[None, :], args.roi_regions, axis=0)  # (36, D)
        ent2vec[eid] = roi
        ok += 1

    with out_pickle.open("wb") as f:
        pickle.dump(ent2vec, f)

    print(f"[Stat] saved={len(ent2vec)}  with_imgs={ok}  missing={missing}")
    print(f"[Done] {out_pickle}")

if __name__ == "__main__":
    main()
