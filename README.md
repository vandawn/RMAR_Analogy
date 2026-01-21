# RMAR_Analogy
Code and datasets for the WWW2026 paper "Relation-Aware Multimodal Analogical Reasoning with Modality Fingerprints and Adaptive Gating"

## MCNet Dataset
The text and image data of the MCNet Dataset can be downloaded through [Google Drive](https://drive.google.com/file/d/1DhIg_S32yIr-h7_W9nxBXkWyK3zH6_T5/view?usp=drive_link).

## Run
### MPT
To evaluate RMAR-MKGFormer, run:
```bash
cd /MPT
bash scripts/run_pretrain_mkgformer_struct.sh
bash scripts/run_finetune_mkgformer_struct.sh
```
### MKGE
To evaluate RMAR-RotatE, run:
```bash
cd /MKGE
bash scripts/run_RMAR.sh
```

## Citations

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3774904.3792481,
  author    = {Ruofan Wang and Zijian Huang and Qiqi Wang and Yuchen Su and Robert Amor and Kaiqi Zhao and Meng-Fen Chiang},
  title     = {Relation-Aware Multimodal Analogical Reasoning with Modality Fingerprints and Adaptive Gating},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  numpages  = {12},
  doi       = {10.1145/3774904.3792481},
  url       = {https://doi.org/10.1145/3774904.3792481},
  year      = {2026}
}
```




