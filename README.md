# MaCo
The code of 'Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning'
Some code is borrowed from MAE, huggingface, and MRM.

## Environmental preparation
```
conda create -n MaCo python=3.8
conda activate MaCo
pip install -r requirements.txt
```
## Links to download datasets
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

- [NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/#:~:text=What%20is%20CheXpert%3F,labeled%20reference%20standard%20evaluation%20sets.)

- [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

- [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

- [MS-CXR](https://aka.ms/ms-cxr)

## Datasets splits
In the directory [DatasetsSplits](DatasetsSplits), we provide dataset splits that may be helpful for organizing the datasets.

We give the train/valid/test splits of [CheXpert](DatasetsSplits/CheXpert), [NIH ChestX-ray](DatasetsSplits/NIH_ChestX-ray), [RSNA Pneumonia](DatasetsSplits/RSNA_Pneumonia), and [SIIM-ACR Pneumothorax](DatasetsSplits/RSNA_Pneumonia).

## Pretraining
Adjust the necessary paths and perform the following code:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 main_pretrain.py;
```

## Fine-tuning of classification
We use NIH ChestX-ray as an example:
```
cd CLS-NIH ChestX-ray
CUDA_VISIBLE_DEVICES=0 python train.py --pretrained_path "./pretrained-model/checkpoint-30.pth";
python test.py --model pretrained-model --gpu 4;
```

## Fine-tuning of segmentation
```
cd 
chmod a+x ft.sh
./ft.sh
chmod a+x test.sh
./test.sh
```
## Zero-shot phase-grounding
```
python eval_Maco
```
