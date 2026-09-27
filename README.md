# Lockheed Martin × UPRM Hackathon 2026 – Plastic Bag Detection

Detector for **plastic bags** in waste-truck camera images, built at the **Lockheed Martin UPRM Hackathon 2026** (Sep 25–27, 2026). Scored with **mAP@50**.

| Model | mAP@50 |
|---|---|
| Faster R-CNN + strong augmentation + multi-scale (validation) | 0.8198 |
| **+ test-time augmentation (validation)** | **0.8238** |
| **Test set** |**0.7987**|

## Approach
- **Transfer learning:** COCO-pretrained **Faster R-CNN (ResNet-50 + FPN v2)** from `torchvision`, fine-tuned on 628 images (1 class: `plastic_bag`).
- **Training:** SGD with warmup and cosine decay, 12 epochs, on an NVIDIA A10G (AWS SageMaker).
- **Strong augmentation:** blur, noise and grayscale, to handle the low-quality cameras where the model was failing.
- **Multi-scale training:** random image sizes from 544 to 800 px.
- **Test-time augmentation:** each image is predicted normal and flipped, and the boxes are merged with Weighted Boxes Fusion.

## How to Run
```bash
pip install -r requirements.txt
pip install ensemble-boxes
```
1. Put the dataset in `data/{training,val}/{image,label}/`, with KITTI-format labels.
2. Run `hackathon.ipynb` from top to bottom: setup → training (~18 min) → analysis → test.

Checkpoints (`models/`) and data are not tracked in git because of their size.

## Collaborators

| <a href="https://github.com/julianlaracuente"><img src="https://github.com/julianlaracuente.png" width="80" alt="julianlaracuente"/></a> | <a href="https://github.com/moisesupr"><img src="https://github.com/moisesupr.png" width="80" alt="moisesupr"/></a> |
|:---:|:---:|
| [@julianlaracuente](https://github.com/julianlaracuente) | [@moisesupr](https://github.com/moisesupr) |

## Acknowledgements
Thanks to **Lockheed Martin** for organizing the hackathon and providing the challenge and starter notebook.
