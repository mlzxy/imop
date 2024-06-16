# One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation

[Paper](https://arxiv.org/abs/2405.13178)         [Website](https://mlzxy.github.io/imop)

## Install Dependencies

Use python 3.9

```bash
pip install -r requirements.txt 
# note that it installs torch==1.13.0
# which is necessary for some Open3D torch api
```

Then install [Open3D](https://www.open3d.org/docs/release/compilation.html) and [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) from source. The tested commits are  `7c0acac0a50293c52d2adb70967e729f98fa5018` and `2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7` for Open3D and Pytorch3D, respectively.

Then install torch_geometric==2.4.0, torch_cluster==1.6.1, and torch_scatter==2.1.1. You may need to download the precompiled wheel https://data.pyg.org/whl/. 


## Evaluation and Training

Download datasets and weights from [https://rutgers.box.com/s/icwvszhcb5jvp8zr33htpqboebk2aupq](https://rutgers.box.com/s/icwvszhcb5jvp8zr33htpqboebk2aupq), extract them as `weights` and `datasets` folders. 


```bash
# Run evaluation on novel tasks
python3 ./eval_novel.py config=configs/eval_novel.yaml

# Run evaluation on base tasks
python3 ./eval_base.py config=configs/eval_base.yaml

# Run training
python3 ./train.py config=configs/train_invariant_region.yaml
python3 ./train.py config=configs/train_{region_match, region_match_fine}.yaml
```

Note this released version simplifies the original implementation by removing the state routing (just using the next key pose), and directly using the RLBench instance mask name to determine the groundtruth of invariant regions. Some clustering heuristics are also used during evaluation to improve the accuracy for certain tasks (details in the [eval_base/novel.py](eval_novel.py)).

The network implementation is in [network.py](network.py) and [geometry_lib.py](geometry_lib.py). The [geometry_lib.py](geometry_lib.py) is a single file implementation for: 

1. A set of primitives of batching point cloud of different sizes (`to_dense_batch / to_flat_batch / ...`)
2. Point transformer (`PointTransformerNetwork`)
3. KNN graph transformer (`KnnTransformerNetwork / make_knn_transformer_layers / ...`)
4. DualSoftmaxMatching (`DualSoftmaxReposition`) for the correspondence-based regression.
5. Utilities such as knn search (`knn/knn_gather/resample`), farthest point sampling (`fps_by_sizes`) and etc. 


## Some comments

I believe region matching is a good idea for solving one-shot manipulation learning. However, many things could be improved. For example: 

1. It would be better to match image regions, instead of point cloud regions. 
2. Region matching inherently contains multi-modality, which needs to be considered from the beginning.  I suggest to checkout the failure cases in the supplementary 
https://mlzxy.github.io/imop/supplementary.pdf.  
3. My initial version is based on object-centric representation (selecting masks as invariant regions). Later I remove the object-centric assumption for a supposely more general setting, but this could be wrong because the object-centric version works better and way more robust.




# Citation

In case this work is useful

```bibtex
@inproceedings{zhang2024oneshot,
      title={One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation}, 
      author={Xinyu Zhang and Abdeslam Boularias},
      booktitle = {Proceedings of Robotics: Science and Systems},
      address  = {Delft, Netherlands},
      year = {2024}
}
```