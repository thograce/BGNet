# BGNet: Boundary-Guided Camouflaged Object Detection (IJCAI 2022)


> **Authors:** 
> Yujia Sun,
> Shuo Wang,
> Chenglizhao Chen,
> and Tian-Zhu Xiang.

## 1. Preface

- This repository provides code for "_**Boundary-Guided Camouflaged Object Detection**_" IJCAI-2022. [![Arxiv Page](https://img.shields.io/badge/Arxiv-2207.00794-red?style=flat-square)](https://arxiv.org/abs/2207.00794)

## 2. Proposed Baseline

### 2.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA Tesla P40 GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n BGNet python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/best/BGNet.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1RQFRgQNtXYfKW4bm5veU4_4ikgPiuEwl/view?usp=sharing).
    
    + downloading Res2Net weights and move it into `./models/res2net50_v1b_26w_4s-3cf99910.pth`[download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `etrain.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `etest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 2.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

If you want to speed up the evaluation on GPU, you just need to use the [efficient tool](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.

Assigning your costumed path, like `method`, `mask_root` and `pred_root` in `eval.py`.

Just run `eval.py` to evaluate the trained model.

> pre-computed maps of BGNet can be found in [download link (Google Drive)](https://drive.google.com/file/d/1vhrAGJI81YAK9YSYgPJer0kxzNEfnRT2/view?usp=share_link).

> pre-computed maps of other comparison methods can be found in [download link (Baidu Pan)](https://pan.baidu.com/s/1dLMqa4tix1gdBN1uWrCPbQ) with Code: yxy9.

## 3. Citation

Please cite our paper if you find the work useful: 

	@inproceedings{sun2022bgnet,
	title={Boundary-Guided Camouflaged Object Detection},
	author={Sun, Yujia and Wang, Shuo and Chen, Chenglizhao and Xiang, Tian-Zhu},
	booktitle={IJCAI},
	pages = "1335--1341",
	year={2022}
	}
