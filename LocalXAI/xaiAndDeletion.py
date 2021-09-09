# !pip install --user timm
# !pip install --user grad-cam
# !pip install --user ttach
# !pip install --user lightning-bolts
# !pip install --user jupyter ipywidgets widgetsnbextension pandas-profiling
# !pip install captum


### IMPORTS ###


import tracemalloc
import os
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import timm
import lime
import math
import sys

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

from PIL import Image
import numpy as np

import tqdm
import pytorch_grad_cam
import cv2

from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised import SimCLR
from torchvision import datasets, transforms
from torchvision.models import resnet50

from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import InputXGradient
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import Deconvolution
from captum.attr import Occlusion
from captum.attr import visualization as viz


### CODE ###

# Set CUDA GPU on cluster from a python script

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[2])

# !jupyter nbextension enable --py widgetsnbextension


# # Load SwaV Model
# weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
# # weight_path = '/home_nfs/stassinsed/Workshop/resnet50_swav_13'
# modelswav = SwAV.load_from_checkpoint(weight_path, strict=True).model
# modelswav.prototypes=nn.Linear(128, 2)
# model_split = "resnet50_swav_13"
# modelswav.load_state_dict(torch.load(model_split))

# modelswav2 = SwAV.load_from_checkpoint(weight_path, strict=True).model
# modelswav2


class Flatten(nn.Module):
    def forward(self, input):
        return input[0]


# Load SimCLR model
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
modelsimclr = nn.Sequential(
    simclr.encoder,
    Flatten(),
    nn.Linear(2048,2))

modelsimclr.load_state_dict(torch.load('resnet50_simclr_5', map_location='cuda'))
model = modelsimclr
model.eval()
model_split = "resnet50_simclr_5"


# Class containing XAI methods used from Captum framework : https://github.com/pytorch/captum
class CaptumMethods:
    method_names = ['GradientShap', 'IntegratedGradients', 'Saliency',
                   'InputXGradient', 'GuidedBackprop', 'GuidedGradCam', 'Deconvolution', 
                   'Occlusion']
    
    def __init__(self):
        self.method = None
        self.method_name = None
        
    
    # Create class of the method_name chosen
    def set_method(self, model, method_name):
        assert method_name in self.method_names, 'method not supported'
        self.method_name = method_name
        self.model = model
        
        if method_name == 'GradientShap':
            self.method = GradientShap(model)
        elif method_name == 'IntegratedGradients':
            self.method = IntegratedGradients(model)
        elif method_name == 'Saliency':
            self.method = Saliency(model)
        elif method_name == 'InputXGradient':
            self.method = InputXGradient(model)
        elif method_name == 'GuidedBackprop':
            self.method = GuidedBackprop(model)
        elif method_name == 'GuidedGradCam':
            self.method = GuidedGradCam(model, model[0].layer4[-1])
        elif method_name == 'Deconvolution':
            self.method = Deconvolution(model)
        elif method_name == 'Occlusion':
            self.method = Occlusion(model)
            
    # Create map of XAI method_name on model with input_
    def attribute(self, input_, **other_params):
        rand_img_dist = torch.cat([input_ * 0, input_ * 1])
        output = self.model(input_)
        output = F.softmax(output, dim=1)
        prediction_score, target = torch.topk(output, 1)
        print(prediction_score, target)
        if self.method_name == 'GradientShap':
            if 'n_samples' not in other_params:
                other_params['n_samples'] = 50
            if 'stdevs' not in other_params:
                other_params['stdevs'] = 0.0001
            if 'baselines' not in other_params:
                other_params['baselines'] = rand_img_dist
        elif self.method_name == 'IntegratedGradients':
            if 'n_steps' not in other_params:
                other_params['n_steps'] = 200
        elif self.method_name == 'Saliency':
            pass
        elif self.method_name == 'InputXGradient':
            pass
        elif self.method_name == 'GuidedBackprop':
            pass
        elif self.method_name == 'GuidedGradCam':
            pass
        elif self.method_name == 'Deconvolution':
            pass
        elif self.method_name == 'Occlusion':
            if 'strides' not in other_params:
                other_params['strides'] = (3, 50, 50)
            if 'sliding_window_shapes' not in other_params:
                other_params['sliding_window_shapes'] = (3,60, 60)
            if 'baselines' not in other_params:
                other_params['baselines'] = 0
        
        return self.method.attribute(inputs=input_, target=target, **other_params)




# # Pytorch Grad CAM

# image_size = [224,224]
# test_data = datasets.ImageFolder(root="/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/NORMAL/",transform= transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             ]))

# img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
# img = Image.open(img_test_path).convert("RGB")
# img = transform.Resize(image_size)
# img = transform.ToTensor()
# input_tensor = transform(img).unsqueeze(0)

# # data_transform = transforms.Compose([
# #             transforms.Resize(image_size),
# #             transforms.ToTensor(),
# #             ])
# # test_image = Image.open("/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg").convert("RGB")
# # input_tensor = model(data_transform(test_image).unsqueeze(0))

# # sor = target_layer(data_transform(test_image).unsqueeze(0))

# image_size = [224,224]
# # model = modelswav
# # model_split = "resnet50_swav_13"
# 
# model = modelsimclr
# model_split = "resnet50_simclr_5"
# 
# model.eval()
#  
# # model = resnet50(pretrained=True)
# # CP_PATH = "save_pneumonia_imagnet_resnet50"
# # model_name = "test_ckpt.pth"
# # model_split = model_name.split(".")[0]
# print(model_split)
# # model = timm.create_model(model_name="resnet50", checkpoint_path = CP_PATH + "/" + model_name)
# 
# if model_split =="resnet50_simclr_5":
#     target_layer = model[0].layer4[-1]
# elif model_split =="resnet50_swav_13":
#     target_layer = model.layer4[-1]
# 
# # input_tensor = # Create an input tensor image for your model..
# 
# # img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/PNEUMONIA/person1_virus_9.jpeg"
# # img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/NORMAL/IM-0009-0001.jpeg"
# # img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/NORMAL/IM-0046-0001.jpeg"
# # img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/NORMAL/NORMAL2-IM-0311-0001.jpeg"
# img_test_path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/NORMAL/NORMAL2-IM-0380-0001.jpeg"
# 
# split = img_test_path.split("/")[-1]
# img_split = split.split(".")[0]
# print(img_split)
# 
# # img = Image.open(img_test_path).convert("RGB")
# 
# # config = resolve_data_config({}, model=model)
# # transform = create_transform(**config)
# 
# # input_tensor = transform(img).unsqueeze(0)
# 
# data_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             ])
# test_image = Image.open(img_test_path).convert("RGB")
# input_tensor = data_transform(test_image).unsqueeze(0)
# 
# # Note: input_tensor can be a batch tensor with several images!
# 
# # Construct the CAM object once, and then re-use it on many images:
# use_cuda = True
# 
# # def modelling(model):
#     
# #     modelling = model
# #     return(modelling)
# 
# cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
# camplus = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=use_cuda)
# # camscore = ScoreCAM(model=model, target_layer=target_layer, use_cuda=False)
# # cam_ablation = AblationCAM(model=model, target_layer=target_layer, use_cuda=False)
# # camx = XGradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
# # cameigen = EigenCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
# camgeigen = EigenGradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
# 
# # If target_category is None, the highest scoring category
# # will be used for every image in the batch.
# # target_category can also be an integer, or a list of different integers
# # for every image in the batch.
# # target_category = 281
# target_category = 1
# 
# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# # aug_smooth = False
# # eigen_smooth = False
# 
# rgb_img = cv2.imread(img_test_path, 1)[:, :, ::-1]
# rgb_img = cv2.resize(rgb_img, (224,224))
# rgb_img = np.float32(rgb_img) / 255
# 
# cmap, rgb_img = saveImg(cam, rgb_img, img_split, model_split, target_category, name="GCAM")
# camplus, rgb_img = saveImg(camplus, rgb_img, img_split, model_split, target_category, name ="GCAM++")
# # saveImg(camscore, rgb_img, img_split, model_split, target_category, name ="ScoreCAM")
# # saveImg(cam_ablation, rgb_img, img_split, model_split, target_category, name ="AblationCAM")
# # saveImg(camx, rgb_img, img_split, model_split, target_category, name ="XGradCAM")
# # saveImg(cameigen, rgb_img, img_split, model_split, target_category, name ="EigenCAM")
# eigen, rgb_img = saveImg(camgeigen, rgb_img, img_split, model_split, target_category, name ="EigenGradCAM")

def saveImg(cam, rgb_img, input_tensor, img_split, model_split, target_category, name, aug_smooth = False, eigen_smooth = False, xai="CAM"):
    '''Compute XAI method heatmap and save it to a certain name.'''
    if xai=="CAM": 
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)
        # cv2 requires BGR encoding
#         cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cam_image = visualization
        
        path = "results/{}".format(img_split)
        if not os.path.exists(path):
            os.mkdir(path)

        cv2.imwrite('results/{}/{}_{}_{}.jpg'.format(img_split, model_split, target_category, name), cam_image)
    elif xai=="Captum":
        grayscale_cam = cam.attribute(input_= input_tensor)
        default_cmap = plt.get_cmap('jet')
        cam_image, axis = viz.visualize_image_attr_multiple(np.transpose(grayscale_cam.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(input_tensor[0].cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True)
        print(cam_image)
        
#         plt.figure()
#         plt.imshow((cam_image, axis))
        
        path = "results/{}".format(img_split)
        if not os.path.exists(path):
            os.mkdir(path)
            
        cam_image.savefig('results/{}/{}_{}_{}.jpg'.format(img_split, model_split, target_category, name))
    else: 
        print("Method not recognized")
        
    return grayscale_cam, rgb_img


def explain(img_test_path, model, methodName):
    '''Explain an image (path) according to a model and a chosen XAI method.'''
    
    image_size = [224,224]
    
    # CHANGE MODEL SPLIT IF ANOTHER MODEL ! 
    model_split = "resnet50_simclr_5"

    # model to eval mode for XAI method computation (gradients...)
    model.eval()
    
    if model_split =="resnet50_simclr_5":
        target_layer = model[0].layer4[-1]
    elif model_split =="resnet50_swav_13":
        target_layer = model.layer4[-1]
    
    # Split to get image name for saving results correctly
    split = img_test_path.split("/")[-1]
    img_split = split.split(".")[0]
    print(img_split)
    
    # Preprocess image to image_size
    data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            ])
    test_image = Image.open(img_test_path).convert("RGB")
    input_tensor = data_transform(test_image).unsqueeze(0)
    
    # Tensor to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device) 
    
    use_cuda = True

    # Listing of Captum method available
    listing =['GradientShap', 'IntegratedGradients', 'Saliency',
                   'InputXGradient', 'GuidedBackprop', 'GuidedGradCam', 'Deconvolution', 
                   'Occlusion']
  
    
    # By default xai = CAM if methodName not in listing
    xai = "CAM"
    if methodName =="GradCAM":
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName=="GradCAM++":
        cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName=="ScoreCAM":
        cam = ScoreCAM(model=model, target_layer=target_layer, use_cuda=False)
    elif methodName=="AblationCAM":
        cam = AblationCAM(model=model, target_layer=target_layer, use_cuda=False)
    elif methodName=="XGradCAM":
        cam = XGradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName=="EigenCAM":
        cam = EigenCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName=="EigenGradCAM":
        cam = EigenGradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName=="LayerCAM":
        cam = LayerCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    elif methodName in listing:
        cam = CaptumMethods()
        cam.set_method(model = model, method_name = methodName)
        xai = "Captum"
        
    # target category refers to the class we want to get an explanation for CAM methods.
    # If it is None it gets the highest probability class.
    target_category = None
    
    # Load the img in cv & rgb for plotting
    rgb_img = cv2.imread(img_test_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224,224))
    rgb_img = np.float32(rgb_img) / 255
    
    # Compute XAI method and save it
    cmap, rgb_img = saveImg(cam, rgb_img, input_tensor, img_split, model_split, target_category, name=methodName, xai=xai)
    
    # return XAI method computation map, rgb image and tensor
    return cmap, rgb_img, input_tensor


# # XAI Evaluation Deletion Method


# Useful when XAI method ndim is 3 to convert it to grayscale for comparison with other XAI methods
def rgb2gray(rgb):
    '''Convert RGB to GrayScale, according to OpenCV convertion.'''
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_indexes(method):
    '''Sort a matrix and return the indexes associated (- to +)
    as well as the original sorting result from numpy sort&argsort.'''
    
    
    
    # Convert to gray if XAI relevance method is in rgb
    if method.ndim>2:
        
        method = method.cpu()
        method = method.detach().numpy()
        method = rgb2gray(method[0])
        
    # Sort and return indices (from a flattened array)

    sortValue = np.sort(method, axis=None)
    sort = np.argsort(method, axis=None)
    
    # Create empty np like
    indices = np.empty_like(sort, dtype = object)

    
    for k in range(len(sort)):
        
        # Compute first index i and second index j
        i = math.floor(sort[k]/len(method[0]))
        j = sort[k] - i*len(method[0])

        # Error case if k differs from i*img_size+j
        # TO DO CHANGE FOR ANY IMG SIZE !!! 
        if(sort[k]!=(i*224+j)):
            print("Error i*j != sort[k]")
            break

        # Copy the indexes to 'indices'
        indices[k] = (i,j)
        
    # Check if min is correct
    if(method[indices[0]]!=method.min()):
        print("Min not correctly found")
#     else: 
#         print("Min good", method[indices[0]], method.min())
        
    # Check if max is correct
    if(method[indices[-1]]!=method.max()):
        print("Max not correctly found")
#     else: 
#         print("Max good", method[indices[-1]], method.max())    
        
    return indices, sort, sortValue


def erase(image, indices, step = 100, order='d'):
    '''Erase image pixels step by step based on indices position array.
    'Indice' positions need to refer to relevance xai results sorted.'''
    
    # Empty list to stock model predictions for each step.
    resultTrack=[]

    print("Img. shape : ", image.shape)
    
#     data_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             ])
# #     test_image = Image.open(img_test_path).convert("RGB")
#     input_tensor = data_transform(image).unsqueeze(0)
    
    # Delete and compute score step by step
    if order=='d':
        print("desc order : delete lowest relevance pixels first")
        # ADD 0 PROBAS ARRAY TO RESULTTRACK HERE !!! 
        ### TO DO ###
        for i in tqdm(range(step)):

#             image[0, :,  indices[i][0], indices[i][1]] = 0 
            image[0, :,  indices[i][0], indices[i][1]] = 0
            
            
#             result = model.predict(image)
            with torch.no_grad():
                out = model(image)
                proba = torch.nn.functional.softmax(out[0],dim=0)
            resultTrack.append(result)
    elif order=='a':
        print("asc order : delete first highest relevant pixels")
        with torch.no_grad():
                out = model(image)
                result = torch.nn.functional.softmax(out[0],dim=0)
                print(result)
                
#         result = model.predict(image)
        resultTrack.append(result.cpu().detach().numpy())
        for i in tqdm.tqdm(range(step)):

#             image[0, indices[-i][0], indices[-i][1]] = 0 
            image[0, :,  indices[i][0], indices[i][1]] = 0

#             result = model.predict(image)
            with torch.no_grad():
                out = model(image)
                result = torch.nn.functional.softmax(out[0],dim=0)
                del out
                torch.cuda.empty_cache()
                
            resultTrack.append(result.cpu().detach().numpy())
        

    return resultTrack, image


# input_tensor.shape
# 
# if torch.cuda.is_available():
#     model.cuda()

# steps = 10000
# (indices, sort, sortValue) = get_indexes(cmap)
# input_tens = input_tensor.cuda()
# if torch.cuda.is_available():
#     model.cuda()
# resultTrack, erased = erase(input_tens, indices, step=steps, order='a')

# # print(resultTrack)
# zero = []
# for i in range(len(resultTrack)):
#     zero.append(resultTrack[i][1])
# 
# plt.figure(figsize=(8,8))
# plt.ylim((0,1))
# plt.plot(zero)
# plt.xlabel("step")
# plt.ylabel("score")
# plt.title("Image prediction score after each deletion")
# plt.show()

# # path = "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/chest_xray/test/{}".format(paths[0])
# # img, orig_img = load_image(path, preprocessing_function=keras.applications.nasnet.preprocess_input, target_size=(shaped[0], shaped[1]))
# 
# if torch.cuda.is_available():
#     model.cuda()
# steps = 1000
# 
# mths = [(cmap, 'GC'), (camplus,'GC++'), (eigen,'EigenGC')]
# plt.figure()
# plt.xlabel("Step")
# plt.ylabel("Score")
# plt.title("Prediction score after each deletion")
# 
# for meth in mths:
#     input_tens = input_tensor.cuda()
#     zero=[]
#     (indices, sort, sortValue) = get_indexes(meth[0])
#     resultTrack, erased = erase(input_tens, indices, step=steps, order='a')
#     for i in range(len(resultTrack)):
#         zero.append(resultTrack[i][1])
#     plt.plot(zero, label=meth[1])
# 
# plt.legend()
# # plt.show()
# plt.savefig("deletionGraph.png")



# Img to load with sys.argv[1]
img = sys.argv[1]


#images = ['chest_xray/test/NORMAL/IM-0016-0001.jpeg']
images = [img]


model = modelsimclr

if torch.cuda.is_available():
    model.cuda()
    
# Number of steps
steps = 1000

# Methods that we want to use
mths = [('LayerCAM'), ('InputXGradient'), ('GuidedBackprop'), ('Deconvolution'), ('Saliency'), ('GradCAM'),('GradCAM++'),('EigenGradCAM') ,('GuidedGradCam')]

# Method to look later (computation or error)
# ('ScoreCAM')('IntegratedGradients')

                   

# base path where the dataset is stored
basePath= "/home_nfs/stassinsed/MA2/Q2/PhD/ChestXRay2017/"

# path = "test/PNEUMONIA/"
# zeros = []

for meth in mths: 
    print( "--- " + str(meth) + " --- ")
    # Number of img we are currently iterating over
    numImg=0

    # Where to stock probabilities result 
    zero=[]

    # for each image (only one in practice due to python script for Mem.Leakage #TO DO)
    for img_path in images: 
        numImg+=1
        
        # Find the class Name (NORMAL / PNEUMONIA)
        className = img_path.split("/")[-2]
        print(img_path, className)
        
        img_path = str(basePath) + str(img_path)
        # Explain the image/model result according to current XAI method(=meth)
        cmap, rgb_img, input_tensor = explain(img_path, model, meth)

        # tensor to cuda
        input_tens = input_tensor.cuda()

        # Get indexes for deleting pixels afterwards
        (indices, sort, sortValue) = get_indexes(cmap)
        # Delete pixels and get probabilites list
        resultTrack, erased = erase(input_tens, indices, step=steps, order='a')

        # Find the highest probability between the two classes
        if resultTrack[0][0]>0.5:
            classe = 0
        elif resultTrack[0][0]<0.5:
            classe = 1
        else: print("PROBA == 0.5, Cas non codé pour l'instant!")

        # Stock the probabilities in list "zero"
        for i in range(len(resultTrack)):
            zero.append(resultTrack[i][classe])
#             zero[i] = zero[i] + resultTrack[i][classe]

        # Get current img name to stok results efficiently
        img = sys.argv[1]
        split = img.split('/')[-1]
        split = split.split(".")[0]
        print(split)

        path = 'results/{}/'.format(meth)
        # If path does not exists create the folder
        if not os.path.exists(path):
            os.mkdir(path)
        f = open( 'results/{}/{}'.format(meth, split) + '_list', 'w')
        
        # for each proba in zero
        l = [str(x) for x in zero]
        for e in l:
            # write in the file
            f.write(e)
            f.write(" ")
        f.close()
#         print(meth)
#         print(zero)
        
        torch.cuda.empty_cache()
#     zero = zero/float(numImg)
#     zeros.append((zero,meth))
#     plt.plot(zero, label=meth)
    
# for meth in mths:
#     explain()
#     input_tens = input_tensor.cuda()
#     zero=[]
#     (indices, sort, sortValue) = get_indexes(meth[0])
#     resultTrack, erased = erase(input_tens, indices, step=steps, order='a')
#     for i in range(len(resultTrack)):
#         zero.append(resultTrack[i][classe])
#     plt.plot(zero, label=meth[1])

# plt.figure()
# plt.xlabel("Step")
# plt.ylabel("Score")
# plt.title("Prediction score after each deletion")
# for (zero, meth) in zeros:
#     plt.plot(zero, label=meth)
# plt.legend()
# # plt.show()
# plt.savefig("deletionGraph.png")   

