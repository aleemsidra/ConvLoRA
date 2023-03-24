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
from dpipe.torch.functional import weighted_cross_entropy_with_logits

from models import UNet2D
from evaluate import dice_score

from save_model import save_model
from evaluate import evaluate_preds_surface_dice
from datetime import datetime
from utils.logger import save_config #, write_info_to_logger
from utils import logger


# def train_model(dataset_train, dataset_val, save_model_to, save_log_to="log_exp", num_epochs=50, device=torch.device("cuda:0"), batch_size=20, initial_lr=0.001, n_channels_out=1):
def train_model( dataset_train, dataset_val, config, suffix, wandb_mode, device=torch.device("cuda:0"), initial_lr=0.001):
    # print('----------------------------------------------------------------------')
    # print('                    Training started')
    # print('----------------------------------------------------------------------')
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    n_channels_out = config.n_channels_out
    

    best_acc = 0
    best_dc = 0


    save_config(config, suffix, folder_time)

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=10, drop_last=True)
    
    
    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=False, num_workers=10, drop_last=False)


    
    wandb_run = wandb.init( project='UDAS', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)


    print("wandb intialized")
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16)
    

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    
    CE_loss = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=config.save_log_to)
    for epoch in range(1, num_epochs + 1):
        # start_time = time.time()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0

        num_steps = 0
        # for i, batch in enumerate(train_loader):
        
        for i, batch in enumerate(train_loader):

            input_samples, gt_samples, _ = batch

   
            if torch.cuda.is_available():
                var_input = input_samples.cuda(device)
                var_gt = gt_samples.cuda(device)
                model = model.cuda()
         
            preds = model(var_input)


            if n_channels_out == 1:
                loss = CE_loss(preds, torch.argmax(var_gt, dim=1))          #understand this
               
                # print("shape", preds.shape, var_gt.shape, type(preds))


                # preds = torch.sigmoid(preds) > 0.5

                
                # dice = evaluate_preds_surface_dice(preds, var_gt, voxel_dim)
                
            train_loss_total += loss.item()
            # train_dice_total += dice.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1
            if epoch  == 1:
                grid_img = vutils.make_grid(input_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu()[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Ground Truth', grid_img, epoch)


        # asd
        train_loss_total_avg = train_loss_total / num_steps
        # train_dice_total_avg = train_dice_total / num_steps

        model.eval()

        val_loss_total = 0.0
        val_dice_total = 0.0

        num_steps = 0

        # print('----------------------------------------------------------------------')
        # print('                    Validation started')
        # print('----------------------------------------------------------------------')

        for i, batch in enumerate(val_loader):

            input_samples, gt_samples, voxel_dim = batch
           
            with torch.no_grad():
                var_input = input_samples.cuda(device)
                var_gt = gt_samples.cuda(device, non_blocking=True)
                # print("validation ...")
                preds = model(var_input)

                if n_channels_out == 1:
                    loss = CE_loss(preds, torch.argmax(var_gt, dim=1))  
                    # dice = evaluate_preds_surface_dice((torch.sigmoid(preds) > 0.5).cpu().numpy(), var_gt.cpu().numpy(), voxel_dim)
        
                 
                    # print("shape", preds.shape, var_gt.shape, type(preds))


                    dice = evaluate_preds_surface_dice(preds, var_gt, voxel_dim)

                    # dice = evaluate_preds_surface_dice((torch.sigmoid(preds) > 0.5).cpu().numpy(), var_gt.cpu().numpy(), voxel_dim)


                    # dice = evaluate_preds_surface_dice(torch.sigmoid(preds) > 0.5, var_gt, voxel_dim)
                    # loss = weighted_cross_entropy_with_logits(preds, var_gt)
                    # dice = dice_score(torch.sigmoid(preds) > 0.5, var_gt)

                # else:
                #     loss = CE_loss(preds, torch.argmax(var_gt, dim=1))          
                #     dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(var_gt, dim=1), n_outputs=n_channels_out)

                val_loss_total += loss.item()
                val_dice_total += dice.item()

              

            num_steps += 1

            if epoch % 30 == 0 or epoch % num_epochs == 0:
                grid_img = vutils.make_grid(input_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu()[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Ground Truth', grid_img, epoch)

        val_loss_total_avg = val_loss_total / num_steps
        val_dice_total_avg = val_dice_total / num_steps

  
        


        print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {train_dice_total_avg}, Valid Loss, {val_loss_total_avg}, Valid DC: {val_dice_total_avg}')
        
        if val_dice_total_avg > best_acc:
            # print("best- epoch start", best_acc)

            # print("prev", best_acc)
            # print("curr", val_dice_total_avg)
            # print("updated", best_acc)
            
            # if best_acc - val_dice_total_avg == 0.005:   #during refinement, and between the string and weak segmentation oyutput
            #     print("early stopping")
            #     break

            best_acc = val_dice_total_avg
            
            # print("best_acc- after updation", best_acc)
            save_model(model, config, suffix, folder_time)

        # if epoch == 1:
        #     break

    

        wandb_run.log({
                        "Epoch": epoch,
                        "Train Loss": train_loss_total_avg,
                        # "Train DC":   train_dice_total_avg,
                        "Valid Loss": val_loss_total_avg,
                        "Valid DC":   val_dice_total_avg

                    })
    
   
    return model
