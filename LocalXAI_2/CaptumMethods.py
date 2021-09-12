import torch

from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import InputXGradient
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import Deconvolution
from captum.attr import Occlusion

class CaptumMethods:
    method_names = ['GradientShap', 'IntegratedGradients', 'Saliency',
                   'InputXGradient', 'GuidedBackprop', 'GuidedGradCam', 'Deconvolution', 
                   'Occlusion']
    
    def __init__(self):
        self.method = None
        self.method_name = None
    
    def set_method(self, model, method_name):
        assert method_name in self.method_names, 'method not supported'
        self.method_name = method_name
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
            
    def attribute(self, input_, target, **other_params):
        rand_img_dist = torch.cat([input_ * 0, input_ * 1])
        
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
