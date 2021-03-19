# JetVLAD tagger

[Identifying Heavy-Flavor Jets Using Vectors of Locally Aggregated](https://arxiv.org/abs/2005.01842)  
Jana Bielčíková, Raghav Kunnawalkam Elayavalli, Georgy Ponimatkin, Jörn H. Putschke, Josef Šivic  
JINST 16 P03017
  
### Dependecies
Create a new conda environment i.e.
```
conda env create -f environment.yml
```
After it is done run `conda activate netvlad-pytorch` and your environment should be ready.

### Data Preparation
To generate dataset used in the paper please refer to [ponimatkin/jet-generator-tagging](https://github.com/ponimatkin/jet-generator-tagging)
repository which provides the code needed to generate the data.   
To run this code on custom data, do the following:
1. Generate a TTree where each separate entry represent a jet, with each branch being an array of entries.
2. Modify the `JetVectorDataset` class in the [utils.py](utils.py) folder by changing correpsonding names, i.e. `fPt - > fX` e.t.c.
 
### Running the code
To train the model, for example, run the following
```
python train.py --train-data /path/to/train/dataset/*.root \
                --test-data /path/to/test/dataset/*.root \
                --val-data /path/to/val/dataset/*.root \
                --variables trkvtx \
                --clusters 33 \
                --depth 2 \
                --signal hf_vs_l \
                --jobid jetvlad_run_hf_vs_l_depth_2_clusters_33
```
Trained model, PR and ROC curves will be saved at `training_runs/jetvlad_run_hf_vs_l_depth_2_clusters_33`.

### Citations
If you find this code useful in your research, please cite the following:
```
@article{jetvlad,
    title = "{Identifying Heavy-Flavor Jets Using Vectors of Locally Aggregated Descriptors}",
    author = {Biel\v{c}\'\i{}kov\'a, Jana and Kunnawalkam Elayavalli, Raghav and Ponimatkin, Georgy and Putschke, J\"orn H. and \v{S}ivic, Josef},
    journal{arXiv:2005.01842},
    year={2020},
}

@article{Arandjelovic18, 
        author={R. {Arandjelović} and P. {Gronat} and A. {Torii} and T. {Pajdla} and J. {Sivic}}, 
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        title={NetVLAD: CNN Architecture for Weakly Supervised Place Recognition},
        year={2018}, 
        volume={40}, 
        number={6}, 
        pages={1437-1451}
}

@article{miech18learning,
  title={Learning a {T}ext-{V}ideo {E}mbedding from {I}ncomplete and {H}eterogeneous {D}ata},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv:1804.02516},
  year={2018},
}

@article{miech17loupe,
  title={Learnable pooling with Context Gating for video classification},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv:1706.06905},
  year={2017},
}


```


