import torch
from torch import nn

import torchvision

from pl_bolts.models.self_supervised import SimCLR, SwAV


def getModels():
    MODEL_DIR_PATH = '../challenge5models/'
    models_dict = {}
    
    # resnet50_simclr_5
    class Flatten(nn.Module):
        def forward(self, input):
            return input[0]

    weight_path = MODEL_DIR_PATH + 'simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    model_resnet50_simclr_5 = nn.Sequential(
        simclr.encoder,
        Flatten(),
        nn.Linear(2048,2)
    )

    model_resnet50_simclr_5.load_state_dict(torch.load(MODEL_DIR_PATH + 'resnet50_simclr_5', map_location='cpu'))
    _ = model_resnet50_simclr_5.eval()
    models_dict['resnet50_simclr_5'] = model_resnet50_simclr_5
    
    # resnet50_swav_13
    weight_path = MODEL_DIR_PATH + 'swav_imagenet.pth.tar'
    model_resnet50_swav_13 = SwAV.load_from_checkpoint(weight_path, strict=True).model
    model_resnet50_swav_13.prototypes=nn.Linear(128, 2)

    model_resnet50_swav_13.load_state_dict(torch.load(MODEL_DIR_PATH + 'resnet50_swav_13', map_location=torch.device('cpu')))
    _ = model_resnet50_swav_13.eval()
    models_dict['resnet50_swav_13'] = model_resnet50_swav_13
    
    # resnet50_simclr_crop_12
    class Flatten(nn.Module):
        def forward(self, input):
            return input[0]

    weight_path = MODEL_DIR_PATH + 'simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    model_resnet50_simclr_crop_12 = nn.Sequential(
        simclr.encoder,
        Flatten(),
        nn.Linear(2048,2)
    )

    model_resnet50_simclr_crop_12.load_state_dict(torch.load(MODEL_DIR_PATH + 'resnet50_simclr_crop_12', map_location='cpu'))
    _ = model_resnet50_simclr_crop_12.eval()
    models_dict['resnet50_simclr_crop_12'] = model_resnet50_simclr_crop_12
    
    # resnet50_swav_crop_10
    weight_path = MODEL_DIR_PATH + 'swav_imagenet.pth.tar'
    model_resnet50_swav_crop_10 = SwAV.load_from_checkpoint(weight_path, strict=True).model
    model_resnet50_swav_crop_10.prototypes=nn.Linear(128, 2)

    model_resnet50_swav_crop_10.load_state_dict(torch.load(MODEL_DIR_PATH + 'resnet50_swav_crop_10', map_location=torch.device('cpu')))
    _ = model_resnet50_swav_crop_10.eval()
    models_dict['resnet50_swav_crop_10'] = model_resnet50_swav_crop_10
    
    return models_dict
