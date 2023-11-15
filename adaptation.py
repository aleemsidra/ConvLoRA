# Refine UNet early features on the Calgary Campinas or M&Ms Dataset


import numpy as np
import wandb
import copy

import torch
import torch.nn as nn

import torchvision.utils as vutils
from utils.logger import save_config 

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


def target_adaptation(dataset_train, dataset_train_dice, dataset_val, adapt, config, suffix, wandb_mode, initial_lr=0.0001, level=3, device=torch.device("cuda:0")):
    
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

    # CE_loss = torch.nn.CrossEntropyLoss()
    # class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0])
    # class_we
    # CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(device)


    class_weights =  dataset_train_dice[0][-1].cuda(device)
    CE_loss =  torch.nn.CrossEntropyLoss(weight=class_weights.cuda(device))

    # class_weights =  dataset_train_dice[0][-1].cuda(device)
    # CE_val_loss =  torch.nn.CrossEntropyLoss(weight=class_weights.cuda(device))

    print('----------------------------------------------------------------------')
    print('                    Accuracy before Adaptation')                                                
    print('----------------------------------------------------------------------')
    with torch.no_grad():

        model.eval()
        avg_train_dice = []
        for img in range(len(dataset_val)):  # looping over all 3D files
            train_samples, _, voxel = dataset_val[img]  # ((((ORIGNAL)))) 
            # train_samples, var_gt, voxel = dataset_val[img]  # Get the ith image, label, and voxel   
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
   
                elif level == 1:
                    # down 1
                    layer_activations_0 = model.init_path(img_slice)
                    layer_activations_1 = model.down1(layer_activations_0)
                    logits_ = features_segmenter(layer_activations_1)
                    prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')
                
                elif level == 2:
                    # down 2
                    layer_activations_0 = model.init_path(img_slice)
                    layer_activations_1 = model.down1(layer_activations_0)
                    layer_activations_2 = model.down2(layer_activations_1)
                    logits_ = features_segmenter(layer_activations_2)
                    prediction = F.interpolate(logits_, scale_factor=4, mode='bilinear')
                
                elif level == 3:
                    # down 3
                    layer_activations_0 = model.init_path(img_slice)
                    layer_activations_1 = model.down1(layer_activations_0)
                    layer_activations_2 = model.down2(layer_activations_1)
                    layer_activations_3 = model.down3(layer_activations_2)
                    logits_ = features_segmenter(layer_activations_3)
                    prediction = F.interpolate(logits_, scale_factor=8, mode='bilinear')
                if level != 4:
                    predictions.append(prediction.squeeze().detach().cpu())


           
            if level == 4:
                preds = var_gt.squeeze()
            
            else: 
                preds = torch.stack(predictions, dim=0)

            stronger_preds = torch.stack(stronger_predictions, dim= 0)
            stronger_predictions.clear()
            stronger_preds_prob = torch.sigmoid(stronger_preds)
           
            if n_channels_out == 1:
                # embed()
                # loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
                train_dice = sdice(torch.sigmoid(preds).numpy() > 0.5,
                                    stronger_preds_prob.numpy() > 0.5,
                                    voxel[img])
                


            else:
                # loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1))     
                loss = CE_loss(preds.cuda(device), torch.argmax(stronger_preds, dim=1).cuda(device))
                # loss = CE_val_loss(preds.cuda(device), torch.argmax(stronger_preds, dim=1).cuda(device))
                train_dice, _ = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)
            
            

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

    elif adapt == "constrained_lora_down1":
        desired_submodules = ['init_path', "down1"]
        model = replace_layers(model, desired_submodules)
        mark_only_lora_as_trainable(model,bias='lora_only')
        
        # embed()
        for name, param in model.init_path.named_parameters():
                if "bn" in name:
                    param.requires_grad = True 
        
        for name, param in model.down1.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True
    
    elif adapt == "constrained_lora_down2":
        desired_submodules = ['init_path', "down1", "down2"]
        model = replace_layers(model, desired_submodules)
        mark_only_lora_as_trainable(model,bias='lora_only')
        
        # embed()
        for name, param in model.init_path.named_parameters():
                if "bn" in name:
                    param.requires_grad = True 
        
        for name, param in model.down1.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True

        for name, param in model.down2.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True
    
    elif adapt == "constrained_lora_down3":
        # desired_submodules = ['init_path', "down1", "down2", "down3"]
        # model = replace_layers(model, desired_submodules)
        # mark_only_lora_as_trainable(model,bias='lora_only')
        
        # embed()
        for name, param in model.init_path.named_parameters():
                if "bn" in name:
                    param.requires_grad = True 
        
        for name, param in model.down1.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True

        for name, param in model.down2.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True

        for name, param in model.down3.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                  param.requires_grad = True


    
    elif adapt == "constrained_da":
        desired_submodules = ['init_path']
        for name, param in model.init_path.named_psarameters():
                    param.requires_grad = True 
    
    
    elif  adapt == "full_lora":
 
        lora_model = copy.deepcopy(model)
        desired_submodules = ['init_path', "down1", "down2", "up3", "up2", "up1"]
        for name, param in lora_model.named_parameters():
            if "out_path" not in name or "shortcut" not in name:
                lora_model = replace_layers(lora_model, desired_submodules)
                mark_only_lora_as_trainable(lora_model,bias='lora_only')
            for name, param in lora_model.named_parameters():
             if "bn" in name or isinstance(name, nn.BatchNorm2d):
                param.requires_grad = True
        lora_model.cuda(device)

    else:
            raise ValueError("Invalid Value ")

    print(f"params to be adapted")  
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)  
                               
    embed()
    model.cuda(device)
    model.load_state_dict(torch.load(config.checkpoint),strict = False) 
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)

    save_config(config, suffix,folder_time)
    print('----------------------------------------------------------------------')
    print('                    Train Loss Calculation')
    print('----------------------------------------------------------------------')
    

    for epoch in range(1, num_epochs + 1):
       
        model.train() # original 
        # lora_model.train()
        model.eval()
        train_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(train_loader):
              
            input_samples, _, _ = batch  # original
            # input_samples, var_gt, _ = batch 
     
            var_input = input_samples.cuda(device)
            # if level == 4: 
            stronger_preds = model(var_input)
            # else:
                # preds = model(var_input)
     
            if level == 0:
                layer_activations = model.init_path(var_input)
                preds = features_segmenter(layer_activations)
            elif level ==1:  # level = 1
                # down 1
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                logits_ = features_segmenter(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

            elif level == 2:
                # down 2
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                layer_activations_2 = model.down2(layer_activations_1)
                logits_ = features_segmenter(layer_activations_2)
                preds = F.interpolate(logits_, scale_factor=4, mode='bilinear')

            elif level == 3:
                # down3
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                layer_activations_2 = model.down2(layer_activations_1)
                layer_activations_3 = model.down3(layer_activations_2)
                logits_ = features_segmenter(layer_activations_3)
                preds = F.interpolate(logits_, scale_factor=8, mode='bilinear')
            
            elif level == 4:
                #full convlora
                # lora_model.train()
                preds = lora_model(var_input)


            if n_channels_out == 1:
                
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                train_loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
               
            else:
                # loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1)) 
      
                # loss = CE_loss(preds, torch.argmax(stronger_preds, dim=1))   
                train_loss = CE_loss(preds.cuda(device), torch.argmax(stronger_preds, dim=1).cuda(device))
                    
            train_loss_total += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps
        num_steps = 0
        
        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        # embed()
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
                        
                    elif level == 1:  # level = 1
     
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = features_segmenter(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')
                    
                    elif level == 2:
                        # down 2
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        layer_activations_2 = model.down2(layer_activations_1)
                        logits_ = features_segmenter(layer_activations_2)
                        prediction = F.interpolate(logits_, scale_factor=4, mode='bilinear')
                    

                    elif level == 3:
                        # down 3
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        layer_activations_2 = model.down2(layer_activations_1)
                        layer_activations_3 = model.down3(layer_activations_2)
                        logits_ = features_segmenter(layer_activations_3)
                        prediction = F.interpolate(logits_, scale_factor=8, mode='bilinear')

                    elif level == 4:
                        prediction = lora_model(img_slice)

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
                    
                else:
                    train_dice, _ = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)


                avg_train_dice.append(train_dice)

                # if epoch % 5 == 0  and wandb_mode == "online":
                #     mask = torch.zeros(size=segmented_volume.shape) 
                #     mask[segmented_volume > 0.5] = 1 
                #     log_images(train_samples, mask.unsqueeze(1), gt_samples, epoch , "Train_dice") 
                    #  log_images(input_samples, torch.argmax(preds, dim=1).cpu().numpy(), torch.argmax(stronger_preds, dim=1), epoch, "Train_Dice", img) 
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

                    elif level ==1 :  # level = 1
                        # down 1
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = features_segmenter(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')
                    
                    elif level == 2:  # level = 1
                        # down2
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        layer_activations_2 = model.down2(layer_activations_1)
                        logits_ = features_segmenter(layer_activations_2)
                        prediction = F.interpolate(logits_, scale_factor=4, mode='bilinear')

                    elif level == 3:  # level = 1
                        # down3
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        layer_activations_2 = model.down2(layer_activations_1)
                        layer_activations_3 = model.down3(layer_activations_2)
                        logits_ = features_segmenter(layer_activations_3)
                        prediction = F.interpolate(logits_, scale_factor=8, mode='bilinear')

                    elif level == 4:
                        prediction = lora_model(img_slice)

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
               
                else:
                    # val_loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1))   
                    val_loss = CE_loss(preds.cuda(), torch.argmax(stronger_preds, dim=1).cuda())      
                    # val_loss = CE_val_loss(preds.cuda(device), torch.argmax(stronger_preds, dim=1).cuda(device)) 
                    val_dice, _ = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)
                     
                total_loss += val_loss.item()
                avg_val_dice.append(val_dice)

                # if epoch % 5 == 0  and wandb_mode == "online":
                #      log_images(input_samples, torch.argmax(preds, dim=1).cpu().numpy(), torch.argmax(stronger_preds, dim=1), epoch, "Train_Dice", img) 
            
            avg_val_dice = np.mean(avg_val_dice)
            val_loss_total_avg = total_loss / len(dataset_train_dice)

            if avg_val_dice > train_dice_total_avg_old:  
               
                train_dice_total_avg_old = avg_val_dice
                print("best_acc- after updation", train_dice_total_avg_old)
                save_model(model, config, suffix, folder_time,  "lora") 
          
            print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')    
            wandb_run.log({
                                "Epoch": epoch,
                                "Train Loss": train_loss_total_avg,
                                "Train DC":   avg_train_dice,
                                "Valid Loss": val_loss_total_avg,
                                "Valid DC":   avg_val_dice, 
                                # "LR": initial_lr, 
                                # "batch_size": batch_size,
                                # "epochs": num_epochs

                            })

        # if epoch == 1:
        #      break
            
    return model
