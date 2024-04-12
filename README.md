# ContextSeg
This is the official repository for the publication [ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention](https://arxiv.org/abs/2311.16682). Our project page: (https://enigma-li.github.io/projects/contextSeg/contextSeg.html)


## Dataset
Dataset can be downloaded from [CreativeSketch](https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html) and [SPG](https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset).

## Requirments

- tensorflow>=2.12.0 
- numpy>=1.23.5

  
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
