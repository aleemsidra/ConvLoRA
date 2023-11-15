import numpy as np
import torch
import wandb
import random
import os
from torch.autograd import Variable

from collections import namedtuple

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim

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


def test(dataset, adapt, config, suffix, wandb_mode, device=torch.device("cuda:0"), initial_lr=0.001):
    # num_epochs = config.num_epochs
    batch_size = config.batch_size
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out

    CE_loss = torch.nn.CrossEntropyLoss()
    save_config(config, suffix,folder_time)
    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)

    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16)

    if adapt == "lora_only":
        print(f"adapt: {adapt}")
        model = replace_layers(model, ["init_path"]) # layers to in LoRA matrices
        print("layers injected")
        model.load_state_dict(torch.load(config.lora_checkpoint.replace("dc_model.pth", "lora_only.pth")), strict = False)
        model.load_state_dict(torch.load(config.base_model_checkpoint ), strict = False)

    elif adapt == "lora:down1":
        print(f"adapt: {adapt}")
        model = replace_layers(model, ["init_path", "down1"]) # layers to in LoRA matrices
        print("layers injected")
        model.load_state_dict(torch.load(config.lora_checkpoint.replace("dc_model.pth", "lora_only.pth")), strict = False)

        model.load_state_dict(torch.load(config.base_model_checkpoint ), strict = False)

    elif adapt == "lora:down2":
        print(f"adapt: {adapt}")
        model = replace_layers(model, ["init_path", "down1", "down2"]) # layers to in LoRA matrices
        print("layers injected")
        model.load_state_dict(torch.load(config.lora_checkpoint.replace("dc_model.pth", "lora_only.pth")), strict = False)

        model.load_state_dict(torch.load(config.base_model_checkpoint ), strict = False)


    elif adapt == "lora:down3":
        print(f"adapt: {adapt}")
        model = replace_layers(model, ["init_path", "down1", "down2", "down3"]) # layers to in LoRA matrices
        print("layers injected")
        model.load_state_dict(torch.load(config.lora_checkpoint.replace("dc_model.pth", "lora_only.pth")), strict = False)

        model.load_state_dict(torch.load(config.base_model_checkpoint ), strict = False)

    elif adapt == "constrained_lora":
        print(f"adapt: {adapt}")
        model = replace_layers(model, ["init_path"]) # layers to in LoRA matrices
        lora_model = torch.load(config.lora_checkpoint)
        filtered_state_dict = {k: v for k, v in lora_model.items() if "init_path" in k}
        model.load_state_dict(torch.load(config.base_model_checkpoint), strict = False)
        model.load_state_dict(filtered_state_dict, strict = False)
    elif adapt == "ada_bn":
        embed()
        lora_model = torch.load(config.lora_checkpoint)
        filtered_state_dict = {k: v for k, v in lora_model.items() if "running_mean" in k or "running_var" in k}
        model.load_state_dict(torch.load(config.base_model_checkpoint), strict = False)
        model.load_state_dict(filtered_state_dict, strict = False)


    elif adapt == "full":
        model.load_state_dict(torch.load(config.lora_checkpoint), strict = False)
    
    else:
        raise ValueError("Invalid Value ")

    if torch.cuda.is_available():
      model = model.cuda()
    model.eval()
    # embed()
    print('----------------------------------------------------------------------')
    print('                    Testing Started...')
    print('----------------------------------------------------------------------')

    with torch.no_grad():
        total_loss = 0.0
        avg_dice = []
    #     print(f'dataset len: {len(dataset)}')
        
    #     data_path = "/home/sidra/Documents/Domain_Apatation/UDAS/src/Data/OpenDataset/"
    #     meta_info = {}
    #     file_path = os.path.join(data_path, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    #     with open(file_path) as f:
    #         file_content = f.readlines()

    #     header = file_content[0].strip().split(',')
    #     header[1] = header[1].replace(' ', '_')
    #     Meta = namedtuple('Meta', header[1:])
    #     for line in file_content[1:]:
    #         sample = line.strip().split(',')
    #         meta_info[sample[1]] = Meta(*sample[1:])



    #     images_path = os.path.join(data_path, 'Testing')
    #     all_files = np.array(sorted(os.listdir(images_path)))
    #     files = []
    #     for f in all_files:
    #         if meta_info[f].Vendor == "A":
    #             files.append(f)
    #     files = np.array(sorted(files))
    #     files = files

    #     print("files")
    
    #     step_size = len(files) // min(10, len(files))
    #     assert (step_size > 0)
    #     end_ = step_size * min(10, len(files))

    # embed()
    # for i in range(0, end_, step_size):
    #     print(f'i: {i}')
    
    for idx in range(len(dataset)):
                # Get the ith image, label, and voxel
        # input_samples, gt_samples, voxel = dataset[i]
        

        input_samples, gt_samples, voxel = dataset[idx]
        print(input_samples.shape)
        slices = []
        for slice_id, img_slice in enumerate(input_samples): # looping over single img
            img_slice = img_slice.unsqueeze(0)
            img_slice = img_slice.to(device)

            preds = model(img_slice)
            slices.append(preds.squeeze().detach().cpu())

        segmented_volume = torch.stack(slices, axis=0)
        slices.clear()

        if n_channels_out ==1: 
            loss = weighted_cross_entropy_with_logits(segmented_volume.unsqueeze(1), gt_samples)
            test_dice = sdice( torch.sigmoid(segmented_volume).numpy() >0.5,
                                gt_samples.squeeze().numpy()>0,
                                voxel[idx])
                            
            # print(test_dice)
            
        else:
            loss = CE_loss(segmented_volume, torch.argmax(gt_samples, dim=1))  
            test_dice, _ =  dice_score(torch.argmax(segmented_volume, dim=1) ,torch.argmax(gt_samples, dim=1), n_outputs=n_channels_out)

            
            # log_images(input_samples, torch.argmax(segmented_volume, dim=1).cpu().numpy(), torch.argmax(gt_samples, dim=1), 50, "Test_DC", idx )
            # print(test_dice)
    
    
        total_loss += loss.item()
        
        avg_dice.append(test_dice)
        # print("test dice" , avg_dice)
  

        print("logging segmented images")
        mask = torch.zeros(size=segmented_volume.shape) 
        mask[torch.sigmoid(segmented_volume) > 0.5] = 1
        log_images(input_samples, mask.unsqueeze(1), gt_samples, 100 , "Test", idx)            
    
               
    total_loss_avg = total_loss / len(dataset)

    final_avg_dice = np.mean(avg_dice)


    return final_avg_dice, total_loss_avg

            





