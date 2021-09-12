# Challenge 6 - Local Explainable Methods

## Content

`CaptumFinal.ipynb`: Contain the average images and attributions of some parts of the dataset

`confusionMatrix.ipynb`: Contain the different confusion matrices of the models

`CaptumMethods.py`: Caller for the explainable methods from Captum

`ImgTransformer.py`: Encapsulate image preprocessing in a dedicated class

`Models.py`: Function that return a dict with used models during challenge

## Datasets

Chest X-Ray Images (Pneumonia): <https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>

## Models

For models were used here. They were trained in the context of the challenge 5.

- `resnet50_simclr_5`: SimCLR as training method, not cropped images as inputs
- `resnet50_swav_13`: SWAV as training method, not cropped images as inputs
- `resnet50_simclr_crop_12`: SimCLR as training method, cropped images as inputs
- `resnet50_swav_crop_10`: SWAV as training method, cropped images as inputs

## Env Setup Instructions

```conda install pytorch torchvision torchaudio cudatoolkit=10.2 captum -c pytorch
conda install -c conda-forge matplotlib pytorch-lightning
pip install lightning-bolts

!jupyter nbextension enable --py widgetsnbextension```
