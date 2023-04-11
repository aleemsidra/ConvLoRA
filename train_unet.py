# Train UNet model on the Calgary Campinas or M&Ms Dataset
# Author: Rasha Sheikh

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
                              shuffle=True, num_workers=0, drop_last=False)
    
    # train_dice_loader = DataLoader(dataset_train_dice, batch_size=config.batch_size,
    #                           shuffle=False, num_workers=0, drop_last=False)
    
    
    # val_loader = DataLoader(dataset_val, batch_size=batch_size,
    #                         shuffle=False, num_workers=0, drop_last=False)

    
    wandb_run = wandb.init( project='UDAS', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16)    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    CE_loss = torch.nn.CrossEntropyLoss()

    print('----------------------------------------------------------------------')
    print('                    Training started')
    print('----------------------------------------------------------------------')

    for epoch in range(1, num_epochs + 1):
    #     # start_time = time.time()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0

        num_steps = 0
        # for i, batch in enumerate(train_loader):
        for i, batch in enumerate(train_loader):
        
            input_samples, gt_samples, voxel_dim = batch
            # print("input_samples shape:", input_samples.shape)
            # asd
            if torch.cuda.is_available():
                var_input = input_samples.cuda(device)
                var_gt = gt_samples.cuda(device)
                model = model.cuda()
          
            preds = model(var_input)
            if n_channels_out == 1:
               loss = weighted_cross_entropy_with_logits(preds, var_gt)

            train_loss_total += loss.item()
            # train_dice_total += dice.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            # embed()
        if epoch % 10 == 0 and wandb_mode == "online": 
                # print("image", i)
                print("logging training image")
                mask = torch.zeros(size=preds.shape) 
                mask[preds > 0.5] = 1
                log_images(input_samples[:4], mask, gt_samples[:4], epoch, "Train")   
                print("breaking")
                      
        train_loss_total_avg = train_loss_total / num_steps
        train_dice_total_avg = train_dice_total / num_steps
        # num_steps = 0
        

        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        
        avg_train_dice = []
        predictions = []
        # print("len", len(dataset_train_dice)) # --> 40 for GE3 
        for img in range(len(dataset_train_dice)):  # looping over all 3D files
            # print("img_id", img)
            # Get the ith image, label, and voxel
            input_samples, gt_samples, voxel = dataset_train_dice[img]
            # print(f"Image shape: {input_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
            slices = []
            
            for slice_id, img_slice in enumerate(input_samples): # looping over single img
                # print("slice_id:", slice_id)
                # imageio.imwrite("/home/sidra/Documents/image.png",input_samples[slice_id].squeeze() )
                # embed()
                img_slice = img_slice.unsqueeze(0)
                with torch.no_grad():
                    img_slice = img_slice.cuda(device)
                preds = model(img_slice)
                # preds = torch.sigmoid(preds) > 0.5
                preds = torch.sigmoid(preds)
                # slices.append(preds.squeeze().detach().cpu().numpy())
                slices.append(preds.squeeze().detach().cpu())
            # embed()
            # segmented_volume = np.stack(slices, axis=0)
            segmented_volume = torch.stack(slices, axis=0)
            # print("slice_id",segmented_volume.shape )
            # embed()
            # dice = sdice(segmented_volume.detach().cpu().numpy() >0.5, gt_samples.squeeze().detach().cpu().numpy()>0, voxel[img])
            dice = sdice(segmented_volume.numpy() >0.5, gt_samples.squeeze().numpy()>0, voxel[img])

            # print("img_id", img, "dice_per_img", dice)
            avg_train_dice.append(dice)

            if epoch % 10 == 0 and img ==0 and wandb_mode == "online":
   
                print("logging train_dice_images")
                mask = torch.zeros(size=segmented_volume[125:129].shape) 
                mask[segmented_volume[125:129] > 0.5] = 1
                log_images(input_samples[125:129], mask.unsqueeze(1), gt_samples[125:129], epoch , "Train_dice") 
            
            # if img == 1:   # remove
            #     break
        
        print("avg_dice", np.mean(avg_train_dice))
        avg_train_dice = np.mean(avg_train_dice)
        
      

    #     embed()
        print('----------------------------------------------------------------------')
        print('                    Val Dice Calculation')
        print('----------------------------------------------------------------------')

        model.eval()
        avg_val_dice = []
        total_loss = 0.0
        slices = []

        for img in range(len(dataset_val)):
            # print("img_id", img)
            # Get the ith image, label, and voxel
            input_samples, gt_samples, voxel = dataset_val[img]
            # print(f"Image shape: {input_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
            slices = []
            for slice_id, img_slice in enumerate(input_samples):
                # print("slice_id:", slice_id)
                # embed()
                img_slice = img_slice.unsqueeze(0)
                with torch.no_grad():
                    img_slice = img_slice.cuda(device)
                preds = model(img_slice)
                # preds = torch.sigmoid(preds) > 0.5
                preds = torch.sigmoid(preds)
                # slices.append(preds.squeeze().detach().cpu().numpy())
                slices.append(preds.squeeze().detach().cpu())
    #         # embed()
    #         # segmented_volume = np.stack(slices, axis=0)
            val_segmented_volume = torch.stack(slices, axis=0)
    #         # print("slice_id",segmented_volume.shape )
    #         # embed()
            if epoch % 10 == 0 and img ==0 and wandb_mode == "online":
                print("logging val_dice_images")
                mask = torch.zeros(size=val_segmented_volume[125:129].shape) 
                mask[val_segmented_volume[125:129] > 0.5] = 1
                log_images(input_samples[125:129], mask.unsqueeze(1), gt_samples[125:129], epoch , "Val_dice")  

            # embed()
            if n_channels_out == 1:
               loss = weighted_cross_entropy_with_logits(val_segmented_volume.unsqueeze(1), gt_samples)
            
            total_loss += loss.item()

            # if img == 1:
            #     break

        val_loss_total_avg = total_loss / len(dataset_val)
        dice = sdice(val_segmented_volume.numpy() >0.5, gt_samples.squeeze().numpy()>0, voxel[img])
        avg_val_dice.append(dice)
        # print("avg_dice",avg_val_dice )

        avg_val_dice  =  np.mean(avg_val_dice)
        print("avg_val_dice", np.mean(avg_val_dice))
    
        print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')
        
        if avg_val_dice > best_acc:
            best_acc = avg_val_dice
            print("best_acc- after updation", best_acc)
            save_model(model, config, suffix, folder_time)


        wandb_run.log({
                        "Epoch": epoch,
                        "Train Loss": train_loss_total_avg,
                        "Train DC":   avg_train_dice,
                        "Valid Loss": val_loss_total_avg,
                        "Valid DC":   avg_val_dice

                    })
        
        # if img == 1:
        #     break 

        # if epoch == 1:
        #     break
    
    return model



