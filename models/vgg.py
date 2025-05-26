import torch
import torch.nn as nn
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, weights_path=None):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=None).features
        
        if weights_path:
            # Load the full VGG19 model state dict
            full_state_dict = torch.load(weights_path, map_location='cpu')
            
            # Extract only the features part and remove the "features." prefix
            features_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('features.'):
                    # Remove "features." prefix to match the Sequential module structure
                    new_key = key[9:]  # Remove "features." (9 characters)
                    features_state_dict[new_key] = value
            
            # Load the processed state dict
            vgg_pretrained_features.load_state_dict(features_state_dict)
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        
        # VGG19 layer indices for the target layers:
        # relu1_1: index 1 (conv1_1=0, relu1_1=1)
        # relu2_1: index 6 (pool1=5, conv2_1=6, relu2_1=7) -> up to 7
        # relu3_1: index 12 (pool2=10, conv3_1=11, relu3_1=12) -> up to 12
        # relu4_1: index 21 (pool3=19, conv4_1=20, relu4_1=21) -> up to 21
        # relu4_2: index 23 (conv4_2=22, relu4_2=23) -> up to 23
        # relu5_1: index 30 (pool4=28, conv5_1=29, relu5_1=30) -> up to 30
        
        for x in range(2):  # up to relu1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):  # up to relu2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):  # up to relu3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):  # up to relu4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 23):  # up to relu4_2
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):  # up to relu5_1
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1_1 = self.slice1(X)  # relu1_1
        h_relu2_1 = self.slice2(h_relu1_1)  # relu2_1
        h_relu3_1 = self.slice3(h_relu2_1)  # relu3_1
        h_relu4_1 = self.slice4(h_relu3_1)  # relu4_1
        h_relu4_2 = self.slice5(h_relu4_1)  # relu4_2 (for content loss)
        h_relu5_1 = self.slice6(h_relu4_2)  # relu5_1
        
        return {
            'relu1_1': h_relu1_1,
            'relu2_1': h_relu2_1,
            'relu3_1': h_relu3_1,
            'relu4_1': h_relu4_1,
            'relu4_2': h_relu4_2,
            'relu5_1': h_relu5_1
        }
