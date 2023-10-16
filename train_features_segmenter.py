# Train FeaturesSegmenter model on the Calgary Campinas or M&Ms Dataset


import numpy as np
import wandb
import torch

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from save_model import save_model

import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits
import torch.nn.functional as F

from models import FeaturesSegmenter

from models import UNet2D
from evaluate import dice_score

from save_model import load_model
from evaluate import sdice
from utils.utils import log_images
from datetime import datetime
from IPython import embed


def train(dataset_train, dataset_train_dice, dataset_val,  config, suffix, wandb_mode, level=1, device=torch.device("cuda:0")):

    num_epochs = config.num_epochs
    batch_size = config.batch_size
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out
    initial_lr = config.lr
    
    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

    best_acc = 0
    print("level", level)
    assert (level <= 1)

    in_channels = 16 * (2 ** level)
    model = FeaturesSegmenter(in_channels=in_channels, out_channels=n_channels_out)
    model.cuda(device)

    unet_model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 
    unet_model = load_model(config,unet_model )   
    unet_model.cuda(device)
    unet_model.eval()
    CE_loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)

    for epoch in range(1, num_epochs + 1):

        print('----------------------------------------------------------------------')
        print('                    Training started')
        print('----------------------------------------------------------------------')

        model.train()
        train_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(train_loader):
            input_samples, gt_samples, _ = batch

            var_input = input_samples.cuda(device)
            var_gt = gt_samples.cuda(device, non_blocking=True)

            if level == 0:
                layer_activations = unet_model.init_path(var_input)
                preds = model(layer_activations)                
             
            else:  # level = 1
     
                layer_activations_0 = unet_model.init_path(var_input)
                layer_activations_1 = unet_model.down1(layer_activations_0)
                logits_ = model(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

            if n_channels_out == 1:
                loss = weighted_cross_entropy_with_logits(preds, var_gt)
            else:
                loss = CE_loss(preds, torch.argmax(var_gt, dim=1))      
                     
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1


            if epoch % 10 == 0  and wandb_mode == "online":
            # if  wandb_mode == "online" and ( epoch % 2 == 0 or epoch ==5):
                # mask = torch.zeros(size=preds.shape) 
                # mask[preds > 0.5] = 1
                # log_images(input_samples, mask, gt_samples, epoch, "Train") 
                log_images(input_samples, torch.argmax(preds, dim=1).cpu().numpy(), torch.argmax(var_gt, dim=1).cpu().numpy(), epoch, "Train_Slice_DC",i ) 

        train_loss_total_avg = train_loss_total / num_steps

        num_steps = 0

        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        with torch.no_grad():
            model.eval()
            val_loss_total = 0.0
            val_dice_total = 0.0
            avg_train_dice = []
            for img in range(len(dataset_train_dice)):  # looping over all 3D files
   
                train_samples, gt_samples, voxel = dataset_train_dice[img]  # Get the ith image, label, and voxel
                slices = []

                for slice_id, img_slice in enumerate(train_samples): # looping over single img             
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)

                    if level == 0:
                        layer_activations = unet_model.init_path(img_slice)
                        preds = model(layer_activations)
                        
                    else:  # level = 1
                        layer_activations_0 = unet_model.init_path(img_slice)
                        layer_activations_1 = unet_model.down1(layer_activations_0)
                        # embed()
                        logits_ = model(layer_activations_1)
                        preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    slices.append(preds.squeeze().detach().cpu())
        
                segmented_volume = torch.stack(slices, dim=0)
                slices.clear()

                if n_channels_out == 1:
                    train_dice = sdice(gt_samples.squeeze().numpy()>0,
                                    torch.sigmoid(segmented_volume).numpy() >0.5,
                                        voxel[img])
                
                else:
                    train_dice =  dice_score(torch.argmax(segmented_volume, dim=1) ,torch.argmax(gt_samples, dim=1), n_outputs=n_channels_out)

                avg_train_dice.append(train_dice)

                if epoch % 10 == 0  and wandb_mode == "online":
                # if  wandb_mode == "online":
        
                    # print("logging train_dice_images")
                    # mask = torch.zeros(size=segmented_volume.shape) 
                    # mask[torch.sigmoid(segmented_volume) > 0.5] = 1 #thresholding
                    # log_images(train_samples, mask.unsqueeze(1), gt_samples, epoch , "Train_dice")
                    log_images(train_samples, torch.argmax(segmented_volume, dim=1).cpu().numpy(), torch.argmax(gt_samples, dim=1), epoch, "Train_DC", img) 

            avg_train_dice = np.mean(avg_train_dice)



        # print('----------------------------------------------------------------------')
        # print('                    Val Dice Calculation')
        # print('----------------------------------------------------------------------')
        
        with torch.no_grad():
            model.eval()
            avg_val_dice = []
            total_loss = 0.0
            val_loss_total_avg = 0.0 
            slices = []

            for img in range(len(dataset_val)):
 
                input_samples, gt_samples, voxel = dataset_val[img] # Get the ith image, label, and voxel
              
                slices = []
                for slice_id, img_slice in enumerate(input_samples):
               
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)

                    if level == 0:
                        layer_activations = unet_model.init_path(img_slice)
                        preds = model(layer_activations)
              
                        
                    else:  # level = 1
                        layer_activations_0 = unet_model.init_path(img_slice)
                        layer_activations_1 = unet_model.down1(layer_activations_0)
                        logits_ = model(layer_activations_1)
                        preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    slices.append(preds.squeeze().detach().cpu())
                val_segmented_volume = torch.stack(slices, dim=0)
                slices.clear()
                       
                if n_channels_out == 1:
                    loss = weighted_cross_entropy_with_logits(val_segmented_volume.unsqueeze(1), gt_samples)
                    val_dice = sdice(gt_samples.squeeze().numpy()>0,
                                    torch.sigmoid(val_segmented_volume).numpy() >0.5,
                                    voxel[img])
                else:
                    loss = CE_loss(val_segmented_volume, torch.argmax(gt_samples, dim=1))
                    val_dice = dice_score(torch.argmax(val_segmented_volume, dim=1) ,torch.argmax(gt_samples, dim=1), n_outputs=n_channels_out)

                total_loss += loss.item()
                avg_val_dice.append(val_dice)

                if epoch % 10 == 0  and wandb_mode == "online":
                # if  wandb_mode == "online":
               
                    # mask = torch.zeros(size=val_segmented_volume.shape) 
                    # mask[torch.sigmoid(val_segmented_volume) > 0.5] = 1
                    # log_images(input_samples, mask.unsqueeze(1), gt_samples, epoch , "Val_dice", img) 
                    log_images(input_samples, torch.argmax(val_segmented_volume, dim=1).cpu().numpy(), torch.argmax(gt_samples, dim=1), epoch, "Val_dice", img) 

            val_loss_total_avg = total_loss / len(dataset_val)
            avg_val_dice  =  np.mean(avg_val_dice)
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

    return model
