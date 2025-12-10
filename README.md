# Unveiling Privacy Risks in the Long Tail: Membership Inference in Class Skewness

This repository contains the implementation of membership inference in long-tailed learning scenarios.

## Environment Setups

We recommend using **Miniconda** to manage the Python environment. Install the following packages in the virtual environment `ltmi`:

```shell
conda create -n ltmi python=3.8.19
conda activate ltmi 

pip instll timm
pip install h5py
pip install seaborn
pip install matplotlib

pip install yacs==0.1.8
pip install ftfy==6.1.1
pip install torch==2.4.0
pip install yacs==0.1.8
pip install ftfy==6.1.1
pip install regex==2022.7.9
pip install torchvision==0.19.0
pip install scikit-learn==1.2.1
pip install tensorboard==2.14.0
```

## Dataset Preparation

Prepare the dataset using `make_subsets.py` in the `make_datasets` folder.

```shell
make_subsets.py:
    --datadir: Path to the folder containing the dataset. If it does not exist, the dataset will be downloaded and saved in this folder.
    --subsetdir: Path to the folder where all subset results will be saved.
    --is_longtail: Whether to use long-tailed distribution. Use 'T' for long-tailed or 'F' for balanced distribution.
    --dataname: Dataset name, such as cifar10.
    --im_ratio: Used when is_longtail is 'T'.
```

## Model Training

Train models using `train.py` in the `prepare_models` folder.

```shell
train.py:
    --datadir: Path to the folder containing the dataset. 
    --result_base_dir: Path to the folder where all training results will be saved.
    --subsetsIdxTrainPath: Path to the training subset index file. 
    --subsetsIdxFullPath: Path to the full subset index file. 
    --dataname: Dataset name, such as cifar10.
    --modelIdx: Model index used to select the corresponding training subset.
```

## Membership Inference

Perform membership inference using `mia.py` in the `mia` folder.

```shell
mia.py:
    --subsetsIdxTrainPath: Path to the training subset index file. 
    --subsetsIdxFullPath: Path to the full subset index file. 
    --logistsPath: Path to the base directory containing logits files.
    --modelIdx: Index of the target model. 
```

## Acknowledgements

Our implementation uses the source code from this repository [lira](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021).
