# Anny-Fit

Official code for **Anny-Fit: All-age Human Mesh Recovery** accepted at CVPR 2026 Findings \
[Laura Bravo-Sánchez](https://laubravo.github.io), [Matthieu Armando](https://europe.naverlabs.com/people_user_naverlabs/matthieu-armando/), [Romain Brégier](https://rbregier.github.io), [Grégory Rogez](https://europe.naverlabs.com/people_user_naverlabs/gregory-rogez/), [Serena Yeung-Levy](https://marvl.stanford.edu/people.html), [Fabien Baradel](https://fabienbaradel.github.io)

[![ArXiv](https://img.shields.io/badge/arXiv-2605.04728-33cb56)](https://arxiv.org/abs/2605.04728)
[![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/pdf/2605.04728.pdf)
[![CVPR 2026 Findings](https://img.shields.io/badge/CVPR%202026-Findings%20Track-blue)](https://cvpr.thecvf.com/)

<p align="center">
<img src="assets/cover.webp" width="100%">
</p>

Our method recovers multi-person 3D human meshes from all ages directly in camera space. By integrating expert semantic, depth, keypoint, and segmentation cues, it improves all-age HMR and enables zero-shot adaptation of adult-only models.


## News
- **206/05/05** — Code released

---

## Installation

```bash
git clone --recurse-submodules https://github.com/naver/anny-fit
bash scripts/install.sh
bash scripts/download_checkpoints.sh
source setup.sh
```

See [INSTALL.md](INSTALL.md) for details and troubleshooting tips.

---

## Demo

Download the pre-processed demo data (images + preprocessing) for running Anny-Fit on sample images:
```bash
wget -O demo_data.tar.gz https://download.europe.naverlabs.com/ComputerVision/AnnyFit/demo_data.tar.gz
tar xzf demo_data.tar.gz
```

Or run the preprocessing from scratch on a folder of images using a Multi-HMR 3D mesh initialization (requires all checkpoints):
```bash
python -m preprocess.build_test_dataset \
    --data_root demo \
    --dataset_name multi_person \
    --preprocess_data \
    --detector detectron2
```

Visualize the preprocessing:
```bash
# Optional: visualize preprocessing for a random sample from a folder
python -m visualize.visualize_preprocessing --data_root demo -n 10

# Fitting with Anny-Fit:
cd annyfit
python optimize.py --config configs/demo/multihmr.yaml
```
---
## License

Code is provided under the terms of this [LICENSE](LICENSE.txt) and accompanying [NOTICE](NOTICE.txt).

---

## Citation
If you find our paper or code useful you can cite our work with:
```
@inproceedings{anny-fit2026,
    title={Anny-Fit: All-age Human Mesh Recovery},
    author={Bravo-S{\'a}nchez, Laura and
            Armando, Matthieu and
            Br{\'e}gier, Romain and 
            Rogez, Gr{\'e}gory and
            Yeung-Levy, Serena and
            Baradel, Fabien
            },
    booktitle={CVPR Findings},
    year={2026}
}
```
---
## Our other works
Check out our other works that made this paper possible:
- [Anny](https://github.com/naver/anny) a differentiable all-age human body model.
- [Anny-One](https://europe.naverlabs.com/research/human-centric-computer-vision/anny-one/) a synthetic dataset of 780K+ multi-person and multi-view images with Anny ground-truth meshes.
- [Multi-HMR](https://github.com/naver/multi-hmr) a regression-based 3D human mesh estimation model.
