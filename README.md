# ContextSeg
This is the official repository for the publication [ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention](https://arxiv.org/abs/2311.16682).

## Dataset
Dataset can be downloaded from [CreativeSketch](https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html) and [SPG](https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html).

## Requirments

- Pytorch>=1.6.0
- pytorch_geometric>=1.6.1
- tensorboardX>=1.9
  
### Training
```
python train_Embed.py
```
```
python train_Segformer.py
```
## Citation
```
@article{wang2023contextseg,
  title={ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention},
  author={Wang, Jiawei and Li, Changjian},
  journal={arXiv preprint arXiv:2311.16682},
  year={2023}
}
```
