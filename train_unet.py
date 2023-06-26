# Train UNet model on the Calgary Campinas or M&Ms Dataset


import numpy as np
import torch
import wandb
import os
from torch.autograd import Variable

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import imageio
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


# def train_model(dataset_train, dataset_val, save_model_to, save_log_to="log_exp", num_epochs=50, device=torch.device("cuda:0"), batch_size=20, initial_lr=0.001, n_channels_out=1):
def train_model( dataset_train,  dataset_train_dice, dataset_val, config, suffix, wandb_mode, device=torch.device("cuda:0"), initial_lr=0.001):

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    num_epochs = config.num_epochs
    batch_size = config.batch_size
    n_channels_out = config.n_channels_out
    

    best_acc = 0
    #best_dc = 0

    save_config(config, suffix, folder_time)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

    
    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16)    
    if torch.cuda.is_available():
      model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    # embed()

    for epoch in range(1, num_epochs + 1):

        print('----------------------------------------------------------------------')
        print('                    Training started')
        print('----------------------------------------------------------------------')

        
        model.train()
        train_loss_total = 0.0
        train_loss_total_avg = 0.0
        

        num_steps = 0
        for i, batch in enumerate(train_loader):
        
            input_samples, gt_samples, voxel_dim = batch
            # print("input_samples shape:", input_samples.shape)

            if torch.cuda.is_available():
                var_input = input_samples.to(device)
                var_gt = gt_samples.to(device)
          
            preds = model(var_input)
            if n_channels_out == 1:
               loss = weighted_cross_entropy_with_logits(preds, var_gt)

            train_loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            # embed()
            # if i ==0:
            #     break

            if epoch % 10 == 0  and wandb_mode == "online": 
            # if wandb_mode == "online": 
                    # print("image", i)
                    # print("logging training image")
                    mask = torch.zeros(size=preds.shape) 
                    mask[preds > 0.5] = 1
                    log_images(input_samples, mask, gt_samples, epoch, "Train")   

                      
        train_loss_total_avg = train_loss_total / num_steps

        num_steps = 0

        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        with torch.no_grad():
            model.eval()
            
            avg_train_dice = []
            for img in range(len(dataset_train_dice)):  # looping over all 3D files
                # print("img_id", img)
                train_samples, gt_samples, voxel = dataset_train_dice[img]  # Get the ith image, label, and voxel
                # print(f"Image shape: {input_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
                slices = []

                for slice_id, img_slice in enumerate(train_samples): # looping over single img             
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
                    preds = model(img_slice)
                    slices.append(preds.squeeze().detach().cpu())
         
                
                segmented_volume = torch.stack(slices, dim=0)
                # embed()
                slices.clear()

                # segmented_volume = torch.sigmoid(segmented_volume)
                train_dice = sdice(gt_samples.squeeze().numpy()>0,
                                   torch.sigmoid(segmented_volume).numpy() >0.5,
                                    voxel[img])

                # print("img_id", img, "dice_per_img", dice)
                avg_train_dice.append(train_dice)
                # print("epoch", epoch, "avg_train_dice", avg_train_dice)
                # embed()
                if epoch % 20 == 0  and wandb_mode == "online":
                # if  wandb_mode == "online":
                    # print("logging train_dice_images")
                    mask = torch.zeros(size=segmented_volume.shape) 
                    mask[torch.sigmoid(segmented_volume) > 0.5] = 1 #thresholding
                    log_images(train_samples, mask.unsqueeze(1), gt_samples, epoch , "Train_dice") 
                    
                # if img == 1:   # remove
                #     break
            
            # print("avg_dice", np.mean(avg_train_dice))
                
            avg_train_dice = np.mean(avg_train_dice)

            # embed()
        print('----------------------------------------------------------------------')
        print('                    Val Dice Calculation')
        print('----------------------------------------------------------------------')
        
        with torch.no_grad():
            model.eval()
            avg_val_dice = []
            total_loss = 0.0
            val_loss_total_avg = 0.0 
            slices = []

            for img in range(len(dataset_val)):
                # print("img_id", img)
                
                input_samples, gt_samples, voxel = dataset_val[img] # Get the ith image, label, and voxel
                # print(f"Image shape: {input_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
                slices = []
                for slice_id, img_slice in enumerate(input_samples):
                    # print("slice_id:", slice_id)
                    # embed()
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
                    preds = model(img_slice)
                    slices.append(preds.squeeze().detach().cpu())
        #         # embed()
     
                val_segmented_volume = torch.stack(slices, dim=0)
                # embed()
                slices.clear()
                loss = weighted_cross_entropy_with_logits(val_segmented_volume.unsqueeze(1), gt_samples)
                total_loss += loss.item()
                                        
                val_dice = sdice(gt_samples.squeeze().numpy()>0,
                                torch.sigmoid(val_segmented_volume).numpy() >0.5,
                                voxel[img])
                avg_val_dice.append(val_dice)
                # embed()
                # print("epoch",epoch, "avg_val_dice", avg_val_dice)
     
                if epoch % 20 == 0 and wandb_mode == "online" :
                # if  wandb_mode == "online":
                    # print("logging val_dice_images")
                    mask = torch.zeros(size=val_segmented_volume.shape) 
                    mask[torch.sigmoid(val_segmented_volume) > 0.5] = 1
                    log_images(input_samples, mask.unsqueeze(1), gt_samples, epoch , "Val_dice", img)  

                # if img == 1:
                #     break

            val_loss_total_avg = total_loss / len(dataset_val)

            avg_val_dice  =  np.mean(avg_val_dice)
            # print("avg_val_dice", np.mean(avg_val_dice))
        
            print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')
            # embed()
            if avg_val_dice > best_acc:
                best_acc = avg_val_dice
                print("best_acc- after updation", best_acc)
                save_model(model, config, suffix, folder_time)

            # embed()
            wandb_run.log({
                            "Epoch": epoch,
                            "Train Loss": train_loss_total_avg,
                            "Train DC":   avg_train_dice,
                            "Valid Loss": val_loss_total_avg,
                            "Valid DC":   avg_val_dice

                        })
        
        # if epoch == 2:
        #     break
    
    return model


