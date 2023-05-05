# Refine UNet early features on the Calgary Campinas or M&Ms Dataset


import numpy as np
import wandb

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from save_model import load_model
from evaluate import sdice
import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from save_model import load_model
from evaluate import sdice
from models import FeaturesSegmenter
from models import UNet2D
from utils.utils import log_images
from datetime import datetime
from IPython import embed
# from calgary_campinas_dataset import CalgaryCampinasDataset
from save_model import save_model



# def train_model(dataset_train, save_model_to, model, features_segmenter, save_log_to="log_exp", num_epochs=50, device=torch.device("cuda:0"), batch_size=20, initial_lr=0.001, stopping_thresh=0.005, level=0, n_channels_out=1):

def train_target(dataset_train, dataset_train_dice, dataset_val, config, suffix, wandb_mode, initial_lr=0.001, level=0, device=torch.device("cuda:0")):
  
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    n_channels_out = config.n_channels_out
    stopping_thresh = config.stopping_thresh
    best_dice = 0
    

    wandb_run = wandb.init( project='UDAS', entity='sidra', name = config['model_net_name'] + "_" + suffix +"_"+ folder_time, mode =  wandb_mode)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    
    in_channels = 16 * (2 ** level)
    features_segmenter = FeaturesSegmenter(in_channels=in_channels, out_channels=n_channels_out)
    features_segmenter.load_state_dict(torch.load(config.head_checkpoint))
    features_segmenter.cuda(device)
    features_segmenter.eval()

    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16) 
    model.load_state_dict(torch.load(config.checkpoint)) 
    model.cuda(device)

    assert (level <= 1)

    for p in model.parameters():
        p.requires_grad = False

    if level == 0:
        for p in model.init_path.parameters():
            p.requires_grad = True
    else:  # level 1
        for p in model.init_path.parameters():
            p.requires_grad = True
        for p in model.down1.parameters():
            p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)



    # with torch.no_grad():
    #     model.eval()
    #     avg_train_dice = []
    #     for img in range(len(dataset_val)):  # looping over all 3D files
    #         # print("img_id", img)
    #         train_samples, gt_samples, voxel = dataset_val[img]  # Get the ith image, label, and voxel
    #         # print(f"Image shape: {train_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
    #         # input_samples, gt_samples, _ = batch
            
    #         stronger_predictions = []
    #         predictions = []
    #         # var_input = train_samples.cuda(device)

    #         for slice_id, img_slice in enumerate(train_samples): # looping over single img             
    #             img_slice = img_slice.unsqueeze(0)
    #             img_slice = img_slice.to(device)
    #             stronger_pred = model(img_slice)
    #             stronger_predictions.append(stronger_pred.squeeze().detach().cpu())
    #             # embed()

    #             if level == 0:
    #                 layer_activations = model.init_path(img_slice)
    #                 prediction = features_segmenter(layer_activations)
        
    #             else:  # level = 1
    #                 layer_activations_0 = model.init_path(img_slice)
    #                 layer_activations_1 = model.down1(layer_activations_0)
    #                 logits_ = model(layer_activations_1)
    #                 prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')

    #             predictions.append(prediction.squeeze().detach().cpu())
                

    #         preds = torch.stack(predictions, dim=0)
    #         # embed()
    #         stronger_preds = torch.stack(stronger_predictions, dim= 0)

    #         # embed()
    #         predictions.clear()
    #         stronger_predictions.clear()
    #         stronger_preds_prob = torch.sigmoid(stronger_preds)
    #         # embed()
    #         train_dice = sdice(stronger_preds_prob.numpy() > 0.5,
    #                             torch.sigmoid(preds).numpy() > 0.5,
    #                             voxel[img])
    #         avg_train_dice.append(train_dice)


        # avg_train_dice = np.mean(avg_train_dice)

    # print('avg starting dice', avg_train_dice)
    # train_dice_total_avg_old = avg_train_dice
    train_dice_total_avg_old = 0
    # embed()
    # train_loss_total_avg = train_loss_total / num_steps
    # print('avg starting loss', train_loss_total_avg)
    
    print('----------------------------------------------------------------------')
    print('                    Train Loss Calculation')
    print('----------------------------------------------------------------------')
    
    # embed()
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
                # loss = weighted_cross_entropy_with_logits(preds, stronger_preds)
            
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            if epoch % 10 == 0  and wandb_mode == "online": 
                # print("image", i)
                print("logging training image")
                mask = torch.zeros(size=stronger_preds.shape) 
                mask[stronger_preds > 0.5] = 1
                log_images(input_samples[:4], mask, gt_samples[:4], epoch, "Train")

            # if i == 1:
            #     break

        train_loss_total_avg = train_loss_total / num_steps
        num_steps = 0
        print('avg train loss', train_loss_total_avg)
        # embed()
        print('----------------------------------------------------------------------')
        print('                    Train Dice Calculation')
        print('----------------------------------------------------------------------')
        with torch.no_grad():
            model.eval()
            avg_train_dice = []
            for img in range(len(dataset_train_dice)):  # looping over all 3D files
                # print("img_id", img)
                train_samples, gt_samples, voxel = dataset_train_dice[img]  # Get the ith image, label, and voxel
                # print(f"Image shape: {train_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
                # input_samples, gt_samples, _ = batch
                
                stronger_predictions = []
                predictions = []
        
                for slice_id, img_slice in enumerate(train_samples): # looping over single img             
                    img_slice = img_slice.unsqueeze(0)
                    img_slice = img_slice.to(device)
                    stronger_pred = model(img_slice)
                    stronger_predictions.append(stronger_pred.squeeze().detach().cpu())
                    # embed()

                    if level == 0:
                        layer_activations = model.init_path(img_slice)
                        prediction = features_segmenter(layer_activations)
        
                    else:  # level = 1
                        layer_activations_0 = model.init_path(img_slice)
                        layer_activations_1 = model.down1(layer_activations_0)
                        logits_ = model(layer_activations_1)
                        prediction = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    predictions.append(prediction.squeeze().detach().cpu())
                
        
                preds = torch.stack(predictions, dim=0)
                # embed()
                stronger_preds = torch.stack(stronger_predictions, dim= 0)

                # embed()
                predictions.clear()
                stronger_predictions.clear()
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                # embed()
                train_dice = sdice(stronger_preds_prob.numpy() > 0.5,
                                torch.sigmoid(preds).numpy() > 0.5,
                                    voxel[img])
                avg_train_dice.append(train_dice)

                if epoch % 10 == 0 and img == 0 and wandb_mode == "online":
        
                    print("logging train_dice_images")
                    mask = torch.zeros(size=stronger_preds_prob[125:129].shape) 
                    mask[stronger_preds_prob[125:129] > 0.5] = 1 #thresholding
                    log_images(train_samples[125:129], mask.unsqueeze(1), gt_samples[125:129], epoch , "Train_dice")
                
                # if img == 2:
                #     break
            
            # embed()
            avg_train_dice = np.mean(avg_train_dice)
            
        # embed()
        print('----------------------------------------------------------------------')
        print('                    Val Dice Calculation')
        print('----------------------------------------------------------------------')

        with torch.no_grad():
            model.eval()
            avg_val_dice = []
            total_loss = 0
            for img in range(len(dataset_val)):  # looping over all 3D files
                # print("img_id", img)
                val_samples, gt_samples, voxel = dataset_val[img]  # Get the ith image, label, and voxel
                # print(f"Image shape: {val_samples.shape}, Label shape: {gt_samples.shape}, Voxel shape: {voxel.shape}")
                # input_samples, gt_samples, _ = batch
                
                stronger_predictions = []
                predictions = []
                # var_input = train_samples.cuda(device)

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
                        logits_ = model(layer_activations_1)
                        preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                    predictions.append(prediction.squeeze().detach().cpu())
        
                preds = torch.stack(predictions, dim=0)
                stronger_preds = torch.stack(stronger_predictions, dim= 0)

                # embed()
                predictions.clear()
                stronger_predictions.clear()
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
        
                total_loss += loss.item()

                val_dice = sdice(stronger_preds_prob.numpy() > 0.5,
                                torch.sigmoid(preds).numpy() > 0.5,
                                voxel[img])


                avg_val_dice.append(val_dice)

                if epoch % 10 == 0 and img == 0 and wandb_mode == "online":
        
                    print("logging train_dice_images")
                    mask = torch.zeros(size=stronger_preds_prob[125:129].shape) 
                    mask[stronger_preds_prob[125:129] > 0.5] = 1 #thresholding
                    log_images(train_samples[125:129], mask.unsqueeze(1), gt_samples[125:129], epoch , "val_dice")

                # if img == 2:
                #     break

            val_loss_total_avg = total_loss / len(dataset_val)
            # embed()
            avg_val_dice  =  np.mean(avg_val_dice)
            # embed()
            print('avg val dice', avg_val_dice)
            dice_diff = avg_val_dice - train_dice_total_avg_old
            print('avg train dice_diff', dice_diff)
            train_dice_total_avg_old = avg_val_dice
            
            print(f'Epoch: {epoch}, Train Loss: {train_loss_total_avg}, Train DC: {avg_train_dice}, Valid Loss, {val_loss_total_avg}, Valid DC: {avg_val_dice}')

            wandb_run.log({
                                "Epoch": epoch,
                                "Train Loss": train_loss_total_avg,
                                "Train DC":   avg_train_dice,
                                "Valid Loss": val_loss_total_avg,
                                "Valid DC":   avg_val_dice

                            })
            
            if abs(dice_diff) < stopping_thresh:
                print("----small dice difference...will stop refinement")
                break
            
        
        # if epoch == 1:
        #     break

    save_model(model, config, suffix, folder_time)
   
  

    return model
