# MaCo
The code of 'Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning'
Some code is borrowed from MAE, huggingface, and MRM.

# Environmental preparation
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

- [COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)

- [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

## Datasets splits
In the directory [DatasetsSplits](DatasetsSplits), we provide dataset splits that may be helpful for organizing the datasets.

We give the train/valid/test splits of [CheXpert](DatasetsSplits/CheXpert), [NIH ChestX-ray](DatasetsSplits/NIH_ChestX-ray), and [RSNA Pneumonia](DatasetsSplits/RSNA_Pneumonia).

For [COVID-19 Image Data Collection](DatasetsSplits/COVID-19_Image_Data_Collection), we randomly split the train/valid/test set 5 times and we provide the images in the [images](DatasetsSplits/COVID-19_Image_Data_Collection/images) directory.

For [SIIM-ACR_Pneumothorax](DatasetsSplits/SIIM-ACR_Pneumothorax), please organize the directories of images and annotations as section 5.1 mentioned according to the given splits.

# Pretraining

# Fine-tuning of classification

# Fine-tuning of segmentation

# Zero-shot phase-grounding
