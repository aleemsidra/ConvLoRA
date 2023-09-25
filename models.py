# Definition of PyTorch models
# Author: Rasha Sheikh

from torch import nn
import torch.nn.functional as F

from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.resblock import ResBlock
from dpipe.layers.conv import PreActivation2d
from dpipe.layers.conv import PreActivationND
from IPython import embed

from LoRA.loralib import layers  as lora


class FeaturesSegmenter(nn.Module):

    def __init__(self, in_channels=16, out_channels=1):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, out_channels, kernel_size=3, padding=1)

    def forward(self, x_):
     
        x = F.relu(self.conv1(x_))
        x = F.relu(self.conv2(x))
        out = self.conv3(x)

        return out


# Shirokikh, Boris, et al. "First U-Net layers contain more domain specific information than the last ones."
# Domain Adaptation and Representation Transfer, and Distributed and Collaborative Learning.
# Springer, Cham, 2020. 117-126.
# https://arxiv.org/abs/2008.07357
# https://github.com/kechua/DART20/blob/master/damri/model/unet.py

class UNet2D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size=3, padding=1, pooling_size=2, n_filters_init=8,
                 dropout=False, p=0.1, return_all_activations=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_size = pooling_size
        n = n_filters_init
        if dropout:
            dropout_layer = nn.Dropout(p)
        else:
            dropout_layer = nn.Identity()

        self.init_path = nn.Sequential(
           nn.Conv2d(n_chans_in, n, self.kernel_size, padding=self.padding, bias=False),
           nn.ReLU(),
           ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding)
        )

        self.shortcut0 = nn.Conv2d(n, n, 1)
    

        self.down1 = nn.Sequential(
            nn.BatchNorm2d(n),
            nn.Conv2d(n, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, 1)

        self.down2 = nn.Sequential(
            nn.BatchNorm2d(n * 2),
            nn.Conv2d(n * 2, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),

            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, 1)

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(n * 4),
            nn.Conv2d(n * 4, n * 8, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            dropout_layer
        )

        self.up3 = nn.Sequential(
           ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 8),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up2 = nn.Sequential(
           ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 4),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up1 = nn.Sequential(
           ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
           ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 2),
            nn.ConvTranspose2d(n * 2, n, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.out_path = nn.Sequential(
           ResBlock2d(n, n, kernel_size=1),
            PreActivation2d(n, n_chans_out, kernel_size=1),
            nn.BatchNorm2d(n_chans_out)
        )
        
        self.return_all_activations = return_all_activations

    


    def forward(self, x):
        # print(" in forward function")

        x0_0 = self.init_path[0](x)
        # embed()
        x0_1 = self.init_path[1](x0_0)
        x0_2 = self.init_path[2](x0_1)
        x0_3 = self.init_path[3](x0_2)
        x0 = self.init_path[4](x0_3)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        skip0 = self.shortcut0(x0)
        skip1 = self.shortcut1(x1)
        skip2 = self.shortcut2(x2)

        x2_up = self.up3(x3)
        x1_up = self.up2(x2_up + skip2)
        x0_up = self.up1(x1_up + skip1)
        x_out = self.out_path(x0_up + skip0)

        if not self.return_all_activations:
            return x_out
        else:
            #            return [x0, x1, x2, x3, x2_up, skip2, x1_up, skip1, x0_up, skip0, x_out]
            #            return [x0_2, x0, skip0, x1, skip1, x2, skip2, x3, x2_up, x1_up, x0_up, x_out]
            return [x0_2, skip0, x0, x1, skip1, x2, skip2, x3, x2_up, x1_up, x0_up, x_out]







def add_lora_layers(model):

    for sub_module in model.children():
        desired_submodules = ["init_path" , "down1"]
        # desired_submodules = ["init_path", "shortcut0", "down1", "shortcut1",
        #                        "down2" , "shortcut2" , "down3",
        #                       "up3", "up2" ,"up1" , "out_path"]
        print(f'desired: {desired_submodules}')
        embed()
        for name, sub_module in model.named_children():
            if name in desired_submodules:
                if isinstance(sub_module, nn.Conv2d):
                    extra_layer = lora.Conv2d(sub_module.in_channels, sub_module.out_channels, sub_module.kernel_size[0], r=2, lora_alpha=2)
                    sub_module.add_module('extra_layer', extra_layer)
                                                            

                if isinstance(sub_module, nn.Sequential):
                    for name, layer in list(sub_module.named_children()):  # Convert to list to avoid modifying the iterator
                            if isinstance(layer, nn.Conv2d):
                                extra_layer = lora.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size[0], r=2, lora_alpha=2)
                                # Modify the specific block by adding the extra_layer
                                sub_module.add_module(name, nn.Sequential(layer, extra_layer))
                                
                            elif isinstance(layer, ResBlock):
                                if isinstance(sub_module, nn.Sequential):
                                    for name, layer in list(sub_module.named_children()):
                                        if isinstance(layer, ResBlock):
                                            if isinstance(layer.conv_path, nn.Sequential):
                                                for i, preactivation_module in enumerate(layer.conv_path):
                                                    if isinstance(preactivation_module, PreActivationND) and hasattr(preactivation_module, 'layer') and isinstance(preactivation_module.layer, nn.Conv2d):
                                                            extra_layer = lora.Conv2d(preactivation_module.layer.out_channels,
                                                                                        preactivation_module.layer.out_channels,
                                                                                        preactivation_module.layer.kernel_size[0],
                                                                                            r=2, lora_alpha=2)
                                                            layer.conv_path[i] = nn.Sequential(preactivation_module, extra_layer)

     
    return model




def replace_layers(model):
        
    
        for name, sub_module in model.named_children():
            desired_submodules = ['init_path'] #, "down1"]
            
            # desired_submodules = ['init_path',"shortcut0", "down1","shortcut1" , "down2"
            #                         , "shortcut2", "down3", "up3" , "up2" ,"up1", "out_path"]
            # print("sub", sub_module)
            if name in desired_submodules:
               
                for name, layer in list(sub_module.named_children()): 
                    #Conv2d
                    if isinstance(layer, nn.Conv2d):
                        setattr(sub_module, name, lora.Conv2d(
                        layer.in_channels,
                        layer.out_channels,
                        kernel_size=layer.kernel_size[0],
                        stride = layer.stride,
                        padding = layer.padding,
                        bias = layer.bias  , 
                        r=2,
                        lora_alpha=2))

                    # ResBlock
                    elif isinstance(sub_module, nn.Sequential):
                      
                        for name, layer in list(sub_module.named_children()):
                            if isinstance(layer, ResBlock):
                                    for i, preactivation_module in enumerate(layer.conv_path):
                                        if isinstance(preactivation_module, PreActivationND) and isinstance(preactivation_module.layer, nn.Conv2d):
                                            setattr(preactivation_module, 'layer', lora.Conv2d(
                                                preactivation_module.layer.in_channels,
                                                preactivation_module.layer.out_channels,
                                                kernel_size=preactivation_module.layer.kernel_size[0],
                                                padding = preactivation_module.layer.padding,
                                               bias = preactivation_module.layer.bias,
                                                r=2,
                                                lora_alpha=2))
                                        
                            elif isinstance(layer,PreActivationND) and isinstance(layer.layer, nn.Conv2d):
                          
                                new_lora_layer = lora.Conv2d(
                                in_channels=layer.layer.in_channels,
                                out_channels=layer.layer.out_channels,
                                kernel_size=layer.layer.kernel_size[0],
                                padding=layer.layer.padding[0],
                                bias=layer.layer.bias is not None,
                                r=2,
                                lora_alpha=2)
                                layer.layer = new_lora_layer
                                        

                    
              
        return model
        
                           