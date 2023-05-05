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
    
    save_config(config, suffix,folder_time)
    wandb_run = wandb.init( project='UDAS', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 
    model.load_state_dict(torch.load(config.checkpoint))
    
    if torch.cuda.is_available():
      model = model.cuda()


    model.eval()

    print('----------------------------------------------------------------------')
    print('                    Testing Started...')
    print('----------------------------------------------------------------------')
        

    with torch.no_grad():
        avg_dice_all = []
        total_loss = 0.0
        
        # indices = list(range(len(dataset)))  # create a list of indices for the dataset
        
        # step_size = len(indices) // min(6, len(indices))  # calculate the step size for iterating over indices
        # end_ = step_size * min(6, len(indices))  # calculate the end index for iterating over indices

        # for i in range(0, end_, step_size):  # looping over all folds
        #     # Select 10 random images from the dataset, starting from index i
        #     indices_chunk = indices[i:i+step_size]
        #     indices_subset = random.sample(indices_chunk, min(len(indices_chunk), 10))
        #     print("indices", indices_subset)
                    
        avg_dice = []
        #     for idx in indices_subset:
        # for idx in range(dataset):
        for idx in range(len(dataset)):
                # Get the ith image, label, and voxel
                input_samples, gt_samples, voxel = dataset[idx]
                
                slices = []
                for slice_id, img_slice in enumerate(input_samples): # looping over single img
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
                    preds = model(img_slice)
                    slices.append(preds.squeeze().detach().cpu())

                segmented_volume = torch.stack(slices, axis=0)
                # embed()
                slices.clear()

                loss = weighted_cross_entropy_with_logits(segmented_volume.unsqueeze(1), gt_samples)
                total_loss += loss.item()
         
                test_dice = sdice(gt_samples.squeeze().numpy()>0,
                                   torch.sigmoid(segmented_volume).numpy() >0.5,
                                    voxel[idx])
                avg_dice.append(test_dice)

                print("logging segmented images")
                mask = torch.zeros(size=segmented_volume.shape) 
                mask[torch.sigmoid(segmented_volume) > 0.5] = 1
                # mask = torch.zeros(size=segmented_volume.shape) 

                log_images(input_samples, mask.unsqueeze(1), gt_samples, 100 , "Test", idx) 
               
                # embed()
            
            # avg_dice_all.append(np.mean(avg_dice))
        # embed()
        total_loss_avg = total_loss / len(dataset)
        final_avg_dice = np.mean(avg_dice)
        # final_avg_dice = np.mean(avg_dice_all)

        return final_avg_dice, total_loss_avg

            





