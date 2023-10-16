# Refine UNet early features on the Calgary Campinas or M&Ms Dataset


import numpy as np
import wandb

import torch
import torch.nn as nn

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from save_model import load_model
from evaluate import sdice
from evaluate import dice_score
import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.resblock import ResBlock
from dpipe.layers.conv import PreActivation2d
from dpipe.layers.conv import PreActivationND
from save_model import load_model
from evaluate import sdice

from models import FeaturesSegmenter
from models import UNet2D
from models import replace_layers

from utils.utils import log_images
from datetime import datetime
from IPython import embed

from save_model import save_model

from LoRA.loralib.utils import mark_only_lora_as_trainable


def target_adaptation(dataset_train, dataset_train_dice, dataset_val, adapt, config, suffix, wandb_mode, initial_lr=0.0001, level=0, device=torch.device("cuda:0")):
    
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out


    wandb_run = wandb.init( project='domain_adaptation', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    # data_loader
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    # Early Feature Segmentor
    in_channels = 16 * (2 ** level)
    features_segmenter = FeaturesSegmenter(in_channels=in_channels, out_channels=n_channels_out)
    features_segmenter.load_state_dict(torch.load(config.head_checkpoint))
    features_segmenter.cuda(device)
    features_segmenter.eval()

    # Base Model
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 

    model.load_state_dict(torch.load(config.checkpoint)) 
    for p in model.parameters():
        p.requires_grad = False
        model.cuda(device)   

    print('----------------------------------------------------------------------')
    print('                    Accuracy before Adaptation')                                                
    print('----------------------------------------------------------------------')
    with torch.no_grad():

        model.eval()
        avg_train_dice = []
        for img in range(len(dataset_val)):  # looping over all 3D files

            train_samples, _, voxel = dataset_val[img]  # Get the ith image, label, and voxel   
            stronger_predictions = []
            predictions = []

            for slice_id, img_slice in enumerate(train_samples): # looping over single img             
                img_slice = img_slice.unsqueeze(0)
                img_slice = img_slice.to(device)
                stronger_pred = model(img_slice)
                stronger_predictions.append(stronger_pred.squeeze().detach().cpu())

                if level == 0:
                        layer_activations = model.init_path(img_slice)
                        prediction = features_segmenter(layer_activations)
   
                else:  # level = 1
     
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = features_segmenter(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                predictions.append(prediction.squeeze().detach().cpu())


            preds = torch.stack(predictions, dim=0)
            stronger_preds = torch.stack(stronger_predictions, dim= 0)
            stronger_predictions.clear()
            stronger_preds_prob = torch.sigmoid(stronger_preds)
           
            if n_channels_out == 1:
                
                loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
                train_dice = sdice(torch.sigmoid(preds).numpy() > 0.5,
                                    stronger_preds_prob.numpy() > 0.5,
                                    voxel[img])

            avg_train_dice.append(train_dice)
        avg_train_dice = np.mean(avg_train_dice)


    print('initial dice', avg_train_dice)
    train_dice_total_avg_old = avg_train_dice

    print('----------------------------------------------------------------------')
    print('                    Adaptation Method')                                                
    print('----------------------------------------------------------------------')
         
    if adapt == "constrained_lora":
        desired_submodules = ['init_path']
        model = replace_layers(model, desired_submodules)
        mark_only_lora_as_trainable(model,bias='lora_only')
        for name, param in model.init_path.named_parameters():
                if "bn" in name:
                    param.requires_grad = True    
    
    
    elif  adapt == "full_lora_adapt":
        for name, param in model.named_parameters():
            if "out_path" not in name or "shortcut" not in name:
                model = replace_layers(model)
                mark_only_lora_as_trainable(model,bias='lora_only')

    else:
            raise ValueError("Invalid Value ")

    print(f"params to be adapted")  
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)  
                               
    
    model.cuda(device)
    model.load_state_dict(torch.load(config.checkpoint),strict = False) 
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)


    print('----------------------------------------------------------------------')
    print('                    Train Loss Calculation')
    print('----------------------------------------------------------------------')
    

    for epoch in range(1, num_epochs + 1):
       
        model.train() 
        train_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(train_loader):
              
            input_samples, _, _ = batch
            var_input = input_samples.cuda(device)
            stronger_preds = model(var_input)
     
            if level == 0:
                layer_activations = model.init_path(var_input)
                preds = features_segmenter(layer_activations)
            else:  # level = 1
    
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                logits_ = features_segmenter(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')
    

            if n_channels_out == 1:
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
               
            else:
                loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1)) 
         
            train_loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps
        num_steps = 0
        
        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')

        with torch.no_grad():
            model.eval()

            avg_train_dice = []
            for img in range(len(dataset_train_dice)):  # looping over all 3D files
    
                train_samples, _, voxel = dataset_train_dice[img]  # Get the ith image, label, and voxel    
                stronger_predictions = []
                predictions = []
            
                for _, img_slice in enumerate(train_samples): # looping over single img    
                                
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
    
                    stronger_pred = model(img_slice)
                    stronger_predictions.append(stronger_pred.squeeze().detach().cpu())      
                    
                    if level == 0:
                        layer_activations = model.init_path(img_slice)
                        prediction = features_segmenter(layer_activations)
                        
                    else:  # level = 1
     
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = features_segmenter(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    predictions.append(prediction.squeeze().detach().cpu())
                
        
                preds = torch.stack(predictions, dim=0)
                stronger_preds = torch.stack(stronger_predictions, dim= 0)
                stronger_predictions.clear()
                predictions.clear()
                stronger_preds_prob = torch.sigmoid(stronger_preds)
          
                if n_channels_out == 1:
     
                    train_dice = sdice(torch.sigmoid(preds).numpy() > 0.5,
                                       stronger_preds_prob.numpy() > 0.5,
                                    
                                        voxel[img])
 
                avg_train_dice.append(train_dice)

            avg_train_dice = np.mean(avg_train_dice)
        
        print('----------------------------------------------------------------------')
        print('                    Val Dice Calculation')
        print('----------------------------------------------------------------------')

        with torch.no_grad():
        
            model.eval()
            avg_val_dice = []
            total_loss = 0

            for img in range(len(dataset_val)):  # looping over all 3D files

                val_samples, _, voxel = dataset_val[img]  # Get the ith image, label, and voxel   # Get the ith image, label, and voxel 
                stronger_predictions = []
                slices = []
                for slice_id, img_slice in enumerate(val_samples): # looping over single img    
                            
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
        
                    stronger_pred = model(img_slice)
                    stronger_predictions.append(stronger_pred.squeeze().detach().cpu())

                    if level == 0:
                        layer_activations = model.init_path(img_slice)
                        prediction = features_segmenter(layer_activations)

                    else:  # level = 1
     
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = features_segmenter(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    predictions.append(prediction.squeeze().detach().cpu())


                preds = torch.stack(predictions, dim=0)
                stronger_preds = torch.stack(stronger_predictions, dim= 0)
                stronger_predictions.clear()
                predictions.clear()
                slices.clear()
                stronger_preds_prob = torch.sigmoid(stronger_preds)

                if n_channels_out == 1:

                    val_loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob.squeeze())
                    val_dice = sdice(torch.sigmoid(preds).numpy() > 0.5,
                                    stronger_preds_prob.numpy() > 0.5,
                                    voxel[img])
                    
        
                total_loss += val_loss.item()
                avg_val_dice.append(val_dice)



            avg_val_dice = np.mean(avg_val_dice)
            val_loss_total_avg = total_loss / len(dataset_train_dice)

            if avg_val_dice > train_dice_total_avg_old:  
               
                train_dice_total_avg_old = avg_val_dice
                print("best_acc- after updation", train_dice_total_avg_old)
                save_model(model, config, suffix, folder_time, epoch ) # True to save lora model
          
            print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')    
            wandb_run.log({
                                "Epoch": epoch,
                                "Train Loss": train_loss_total_avg,
                                "Train DC":   avg_train_dice,
                                "Valid Loss": val_loss_total_avg,
                                "Valid DC":   avg_val_dice

                            })
            
    return model
