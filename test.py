import numpy as np
import torch
import wandb
import random
import os
from torch.autograd import Variable

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import imageio
from save_model import load_model
from dpipe.torch.functional import weighted_cross_entropy_with_logits

from utils.utils import log_images
from IPython import embed

from models import UNet2D
from models import replace_layers

from evaluate import dice_score

from save_model import save_model
from evaluate import evaluate_preds_surface_dice, sdice
from datetime import datetime
from utils.logger import save_config #, write_info_to_logger
from utils import logger


def test(dataset, config, suffix, wandb_mode, device=torch.device("cuda:0"), initial_lr=0.001):
    # num_epochs = config.num_epochs
    batch_size = config.batch_size
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out
    CE_loss = torch.nn.CrossEntropyLoss()
    save_config(config, suffix,folder_time)
    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 
    print("model structure loaded")
    embed()
 
    model = replace_layers(model)
    print("layers replaced")

    model.load_state_dict(torch.load(config.lora_checkpoint), strict= False)
    print("lora weights")

    model.load_state_dict(torch.load(config.base_model_checkpoint), strict= False)
    print("lora +base model weights")

    # embed()
    if torch.cuda.is_available():
      model = model.cuda()


    model.eval()

    print('----------------------------------------------------------------------')
    print('                    Testing Started...')
    print('----------------------------------------------------------------------')
        

    with torch.no_grad():
        avg_dice_all = []
        total_loss = 0.0
        
        avg_dice = []
        print(f'dataset len: {len(dataset)}')
        for idx in range(len(dataset)):
                # Get the ith image, label, and voxel
                input_samples, gt_samples, voxel = dataset[idx]
                
                slices = []
                for slice_id, img_slice in enumerate(input_samples): # looping over single img
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
                    preds = model(img_slice)
                    # embed()
                    slices.append(preds.squeeze().detach().cpu())

                segmented_volume = torch.stack(slices, axis=0)
                # embed()
                slices.clear()

                if n_channels_out ==1: 
                    loss = weighted_cross_entropy_with_logits(segmented_volume.unsqueeze(1), gt_samples)
                    test_dice = sdice(gt_samples.squeeze().numpy()>0,
                                   torch.sigmoid(segmented_volume).numpy() >0.5,
                                    voxel[idx])
                else:
                    loss = CE_loss(segmented_volume, torch.argmax(gt_samples, dim=1))  
                    test_dice =  dice_score(torch.argmax(segmented_volume, dim=1) ,torch.argmax(gt_samples, dim=1), n_outputs=n_channels_out)

                total_loss += loss.item()
                avg_dice.append(test_dice)

                print("logging segmented images")
                mask = torch.zeros(size=segmented_volume.shape) 
                mask[torch.sigmoid(segmented_volume) > 0.5] = 1
               
                log_images(input_samples, mask.unsqueeze(1), gt_samples, 100 , "Test", idx)            
                # log_images(input_samples, torch.argmax(segmented_volume, dim=1).cpu().numpy(), torch.argmax(gt_samples, dim=1), 50, "Test_DC", idx )
               


        total_loss_avg = total_loss / len(dataset)
        final_avg_dice = np.mean(avg_dice)
   

        return final_avg_dice, total_loss_avg

            





