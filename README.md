# ContextSeg
This is the official repository for the publication [ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention](https://arxiv.org/abs/2311.16682).
This paper presents ContextSeg - a simple yet highly effective approach to tackling this problem with two stages. In the first stage, to better encode the shape and positional information of strokes, we propose to predict an extra dense distance field in an autoencoder network to reinforce structural information learning. In the second stage, we treat an entire stroke as a single entity and label a group of strokes within the same semantic part using an auto-regressive Transformer with the default attention mechanism. By group-based labeling, our method can fully leverage the context information when making decisions for the remaining groups of strokes. 

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
