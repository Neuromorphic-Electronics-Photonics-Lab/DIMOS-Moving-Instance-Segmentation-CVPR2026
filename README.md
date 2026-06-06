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

> **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1xCFrJ_KrAwNrLH0-ugsfTQ?pwd=1234) (code: 1234) | [OneDrive](https://1drv.ms/f/c/850084079b43c32d/IgAvrIY9Q_0fTrVJX40nEm_RAS6abFl2_m0aOHU25vo9koc?e=KdmfTs)

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

> **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1xCFrJ_KrAwNrLH0-ugsfTQ?pwd=1234) (code: 1234) | [OneDrive](https://1drv.ms/f/c/850084079b43c32d/IgAvrIY9Q_0fTrVJX40nEm_RAS6abFl2_m0aOHU25vo9koc?e=KdmfTs)

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
| MouseSIS  | [Download](https://pan.baidu.com/s/14hxGJoKmyeSdlYTjYKYIfQ?pwd=1234) (code: 1234) | [Download](https://1drv.ms/f/c/850084079b43c32d/IgB4zS-Z9iWQSZEOAtWn9T5-Afy_9w-u5JKS1LsyTPamwSE?e=uRVR7m) |
| SEVD      | [Download](https://pan.baidu.com/s/14hxGJoKmyeSdlYTjYKYIfQ?pwd=1234) (code: 1234) | [Download](https://1drv.ms/f/c/850084079b43c32d/IgB4zS-Z9iWQSZEOAtWn9T5-Afy_9w-u5JKS1LsyTPamwSE?e=uRVR7m) |

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
  title     = {DIMOS: Disentangling Instance-level Moving Object Segmentation},
  author    = {Huang, Hongxiang and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {to appear}
}
```

---

## License

> [TBA]

---

## Acknowledgement

This work is developed at HKUST(GZ).

---

## Contact

If you have any questions, feel free to contact:

- Hongxiang Huang ([hhuang516@connect.hkust-gz.edu.cn](mailto:hhuang516@connect.hkust-gz.edu.cn))
