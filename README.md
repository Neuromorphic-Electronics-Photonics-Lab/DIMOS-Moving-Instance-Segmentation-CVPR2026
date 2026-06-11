# DIMOS: Disentangling Instance-level Moving Object Segmentation

Official PyTorch implementation of:

> **DIMOS: Disentangling Instance-level Moving Object Segmentation**
> *CVPR 2026 (to appear)*

---

## Overview

DIMOS is a novel framework for **moving instance segmentation** that jointly disentangles **appearance** and **motion** representations from both image and event modalities, and aligns them through multi-granularity cross-modal learning.

## Key Ideas

- Dual disentangling of appearance and motion
- Cross-modal alignment between image and event data
- Cross-type interaction for robust fusion
- Task-specific decoders for instance-level segmentation

---

## Installation

```bash
# Create conda environment
conda create -n dimos python=3.9 -y
conda activate dimos

# Install PyTorch (see https://pytorch.org for your CUDA version)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install mmdetection ecosystem
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==3.3.0
pip install mmengine==0.10.7

# Install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### MouseSIS

> **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1xCFrJ_KrAwNrLH0-ugsfTQ?pwd=1234) (code: 1234) | [OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/hhuang516_connect_hkust-gz_edu_cn/IgCRFn9bviSvR7Wx1BcPIp_SAcdfldbQ0425s2ZooNUKTeQ?e=ssO65X)

Organize the dataset as follows:

```
datasets/MouseSIS/
├── emos_preprocess/
│   ├── train/
│   │   └── seq*/
│   │       ├── 00001.hdf5
│   │       ├── 00002.hdf5
│   │       ├── ...
│   │       └── valid.txt
│   └── eval/
│       └── seq*/
│           ├── ...
│           └── valid.txt
└── bbox_gt/
    ├── train.json
    └── val.json
```

### SEVD

> **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1xCFrJ_KrAwNrLH0-ugsfTQ?pwd=1234) (code: 1234) | [OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/hhuang516_connect_hkust-gz_edu_cn/IgCRFn9bviSvR7Wx1BcPIp_SAcdfldbQ0425s2ZooNUKTeQ?e=ssO65X)

Organize the dataset as follows:

```
datasets/sevd/
├── emos_preprocess/
│   ├── train/
│   │   └── seq*/
│   │       ├── 00001.hdf5
│   │       ├── 00002.hdf5
│   │       ├── ...
│   │       └── valid.txt
│   └── eval/
│       └── seq*/
│           ├── ...
│           └── valid.txt
└── bbox_gt/
    ├── train.json
    └── val.json
```

### EVIMO

> Download link: [TBA]
> Training and inference instructions for EVIMO will be added soon.

---

## Model Weights

Pretrained weights for MouseSIS and SEVD are available via the following links (EVIMO weights will be released at a later date):

| Dataset   | Baidu Netdisk | OneDrive |
|-----------|--------------|----------|
| MouseSIS  | [Download](https://pan.baidu.com/s/14hxGJoKmyeSdlYTjYKYIfQ?pwd=1234) (code: 1234) | [Download](https://hkustgz-my.sharepoint.com/:f:/g/personal/hhuang516_connect_hkust-gz_edu_cn/IgANB0tRdoWQRb8ZR5gVYZqXAQRLjwqzkdK_UoE9rOPoo90?e=kxsacP) |
| SEVD      | [Download](https://pan.baidu.com/s/14hxGJoKmyeSdlYTjYKYIfQ?pwd=1234) (code: 1234) | [Download](https://hkustgz-my.sharepoint.com/:f:/g/personal/hhuang516_connect_hkust-gz_edu_cn/IgANB0tRdoWQRb8ZR5gVYZqXAQRLjwqzkdK_UoE9rOPoo90?e=kxsacP) |

Place the downloaded checkpoint files under `logs/<run_name>/ckpts/`.

---

## Training

### MouseSIS

```bash
python train.py -c conf/mousesis.yaml
```

Training logs and checkpoints are saved under `logs/` by default. The configuration uses 400K training steps with batch size 16 on a single GPU.

### SEVD

```bash
python train.py -c conf/sevd.yaml
```

SEVD training uses 800K steps with batch size 16.

### EVIMO

> [TBA]

### Resume Training

```bash
python train.py -c conf/mousesis.yaml --weights logs/<run_name>/ckpts/step-xxxxx.pt --resume
```

---

## Inference / Evaluation

### MouseSIS

```bash
python eval.py -c conf/mousesis.yaml --weights <path_to_checkpoint>.pt --gpus 1 --batch_size 1
```

### SEVD

```bash
python eval.py -c conf/sevd.yaml --weights <path_to_checkpoint>.pt --gpus 1 --batch_size 1
```

### EVIMO

> [TBA]

---

## Configuration

The project uses [OmegaConf](https://github.com/omeliost/omegaconf) for configuration management.

- `conf/default.yaml` — Base configuration shared across datasets.
- `conf/mousesis.yaml` — MouseSIS-specific override (inherits from `default.yaml`).
- `conf/sevd.yaml` — SEVD-specific override (inherits from `default.yaml`).

You can customize training parameters (batch size, learning rate, loss weights, etc.) by modifying the respective YAML files.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{huang2026dimos,
  title={DIMOS: Disentangling Instance-level Moving Object Segmentation},
  author={Huang, Hongxiang and Ren, Hongwei and Lin, Xiaopeng and Huang, Yulong and Xie, Zeke and Cheng, Bojun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={39806--39816},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgement

This work is developed at HKUST(GZ). We thank the authors of [EvInsMOS](https://github.com/danqu130/EvInsMOS) for their open-source codebase, which our implementation builds upon for several foundational modules.

---

## Contact

If you have any questions, feel free to contact:

- Hongxiang Huang ([hhuang516@connect.hkust-gz.edu.cn](mailto:hhuang516@connect.hkust-gz.edu.cn))
