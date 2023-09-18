
import numpy as np
import wandb

import torch
import copy
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from save_model import load_model
from evaluate import sdice
from evaluate import dice_score
import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from save_model import load_model
from evaluate import sdice

from models import FeaturesSegmenter
from models import UNet2D
from models import replace_layers

from utils.utils import log_images
from datetime import datetime
from IPython import embed
# from calgary_campinas_dataset import CalgaryCampinasDataset
from save_model import save_model

from LoRA.loralib.utils import mark_only_lora_as_trainable


def mix_labels(dataset_train, dataset_train_dice, dataset_val, config, suffix, wandb_mode, add_lora = False, initial_lr=0.001, level=0, device=torch.device("cuda:0")):
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    alpha = config.alpha
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out

    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

    # Base Model
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 
    model.load_state_dict(torch.load(config.checkpoint)) 
    model.cuda(device)
    model.eval()
    # print("original model")
    # embed()

    # LoRA Model
    for p in model.parameters():
        p.requires_grad = False

    if level == 0 and  not add_lora:
        for p in model.init_path.parameters():
            p.requires_grad = True

    elif level ==0 and add_lora:
        # Clone the base model
        lora_model = copy.deepcopy(model)
        lora_model = replace_layers(lora_model)
        lora_model.load_state_dict(torch.load(config.checkpoint), strict = False) 
        mark_only_lora_as_trainable(lora_model,bias='lora_only')
        lora_model.cuda(device)
    print("lora_model")    
    # embed()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)

    # Initial Dice Calculation
    # embed()
    with torch.no_grad():

        model.eval()
        avg_train_dice = []
        for img in range(len(dataset_val)):  # looping over all 3D files

            train_samples, gt_samples, voxel = dataset_val[img]  # Get the ith image, label, and voxel   
            stronger_predictions = []
            # predictions = []

            for slice_id, img_slice in enumerate(train_samples): # looping over single img             
                img_slice = img_slice.unsqueeze(0)
                img_slice = img_slice.to(device)
                stronger_pred = model(img_slice)
                stronger_predictions.append(stronger_pred.squeeze().detach().cpu())
 
            stronger_preds = torch.stack(stronger_predictions, dim= 0)
            stronger_predictions.clear()
            stronger_preds_prob = torch.sigmoid(stronger_preds)
            # embed()
         
            train_dice = sdice(gt_samples.squeeze().numpy()>0,
                                stronger_preds_prob.numpy() > 0.5,
                                voxel[img])

            avg_train_dice.append(train_dice)

        # embed()
        avg_train_dice = np.mean(avg_train_dice)

    print('initial dice', avg_train_dice)
    train_dice_total_avg_old = avg_train_dice

    print('----------------------------------------------------------------------')
    print('                    Train Loss Calculation')
    print('----------------------------------------------------------------------')
    for epoch in range(1, num_epochs + 1):
        
        alpha -= alpha / num_epochs
        print(f"epoch: {epoch}, alpha: {alpha}")
        
        model.eval()
        lora_model.train()
        train_loss_total = 0.0
 
        num_steps = 0
        for i, batch in enumerate(train_loader):
              
            input_samples, _ , _ = batch
            var_input = input_samples.cuda(device)
            model_preds = model(var_input)  
            lora_preds = lora_model(var_input)  
            # embed()
            # gt_samples = alpha * model_preds + (1 - alpha) * lora_preds  # ground truth
            gt_samples = alpha * torch.sigmoid(model_preds) + (1 - alpha) * torch.sigmoid(lora_preds) 
         
            loss = weighted_cross_entropy_with_logits(lora_preds, gt_samples)
    
            train_loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1
      

        train_loss_total_avg = train_loss_total / num_steps
        num_steps = 0
        print('avg train loss', train_loss_total_avg)
  

        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        with torch.no_grad():
            
            model.eval()
            lora_model.eval()
            avg_train_dice = []
            for img in range(len(dataset_train_dice)):  # looping over all 3D files
    
                train_samples, _ , voxel = dataset_train_dice[img]  # Get the ith image, label, and voxel    
                train_predictions = []
                train_lora_predictions = []
                
            
                for _, img_slice in enumerate(train_samples): # looping over single img    
                                
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)

                    model_preds = model(img_slice)
                    lora_preds = lora_model(img_slice)

                    train_predictions.append(model_preds.squeeze().detach().cpu())
                    train_lora_predictions.append(lora_preds.squeeze().detach().cpu())
                    
                    # del stronger_pred 
                
                preds = torch.stack(train_predictions, dim= 0)
                lora_preds = torch.stack(train_lora_predictions, dim= 0)
                train_predictions.clear()
                train_lora_predictions.clear()
                
                # stronger_preds_prob = torch.sigmoid(stronger_preds)

                gt_samples = alpha * preds + (1 - alpha) * lora_preds


                if n_channels_out == 1:
                    train_dice = sdice(gt_samples.numpy() > 0.5,
                                       torch.sigmoid(lora_preds).numpy() > 0.5,
                                       voxel[img])
            
                avg_train_dice.append(train_dice)

     
            avg_train_dice = np.mean(avg_train_dice)
            
        # embed()
        print('----------------------------------------------------------------------')
        print('                    Val Dice Calculation')
        print('----------------------------------------------------------------------')

        with torch.no_grad():
            model.eval()
            lora_model.eval()
            avg_val_dice = []
            total_loss = 0

            for img in range(len(dataset_val)):  # looping over all 3D files

                val_samples, _ , voxel = dataset_val[img]  # Get the ith image, label, and voxel   # Get the ith image, label, and voxel 
                val_model_predictions = []
                val_lora_predictions = []
                slices = []
                for slice_id, img_slice in enumerate(val_samples): # looping over single img    
                            
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)

                    model_pred = model(img_slice)
                    lora_pred = lora_model(img_slice)

                    val_model_predictions.append(model_pred.squeeze().detach().cpu())
                    val_lora_predictions.append(lora_pred.squeeze().detach().cpu())

                preds = torch.stack(val_model_predictions, dim= 0)
                lora_preds = torch.stack(val_lora_predictions, dim= 0)
                val_model_predictions.clear()
                val_lora_predictions.clear()

                # gt_samples = alpha * preds + (1 - alpha) * lora_preds
                gt_samples = alpha * torch.sigmoid(preds) + (1 - alpha) * torch.sigmoid(lora_preds) 
                loss = weighted_cross_entropy_with_logits(lora_preds, gt_samples)
                total_loss += loss.item()

                val_dice = sdice(gt_samples.numpy() > 0.5,
                                       torch.sigmoid(lora_preds).numpy() > 0.5,
                                       voxel[img])
            
                avg_val_dice.append(val_dice)

     
            avg_val_dice = np.mean(avg_val_dice)

            val_loss_total_avg = total_loss / len(dataset_val)

            if avg_val_dice > train_dice_total_avg_old:   
                train_dice_total_avg_old = avg_val_dice
                print("best_acc- after updation", train_dice_total_avg_old)
                save_model(lora_model, config, suffix, folder_time)


            print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')

            
            wandb_run.log({
                                "Epoch": epoch,
                                "Train Loss": train_loss_total_avg,
                                "Train DC":   avg_train_dice,
                                "Valid Loss": val_loss_total_avg,
                                "Valid DC":   avg_val_dice

                            })
        
    return model