import numpy as np
import random
import os 

# from requests import request
import torch
from torch.utils.data import DataLoader
import time
from calgary_campinas_dataset import CalgaryCampinasDataset , cc359_3d_volume, cc359_refine
from mms_dataset import MMSDataset, mms_3d_volume
from utils.utils import process_config, check_config_dict
from evaluate import predict_sub, dice_score
from train_unet import train_model
from train_features_segmenter import train
from refine_features import train_target
from psuedo_labels import mix_labels
import matplotlib.pyplot as plt
from test import test

import argparse
from IPython import embed
import torch
# from lora_train import train_model

def main(args, now, suffix, wandb_mode):
    # torch.backends.cudnn.deterministic = True
    # Initializing seeds and preparing GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)
   
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))

    if args.site.isdecimal():
        args.site = int(args.site)

    if not args.test :

        print('----------------------------------------------------------------------')
        print('                    Loading Data ...')
        print('----------------------------------------------------------------------')
        
        print(f"step: {args.step}, data: {args.data}")

        # if args.step == "base_model" or args.step == "feature_segmentor":
        if args.step != "refine" and args.step != "lora":
            if args.data == "cc359":
                # if args.step == "base_model" or args.step == "feature_segmentor":
                    train_data = CalgaryCampinasDataset(config,  args.site)
                    train_dice_data = cc359_3d_volume(config, args.site)
                    val_data = cc359_3d_volume(config, args.site,train= False)
                
            elif args.data == "mms":
                    train_data = MMSDataset(config, args.site)
                
                    train_dice_data = mms_3d_volume(config, args.site)
                    val_data = mms_3d_volume(config, args.site, train= False)
                    
                    print(f'train: {len(train_data)}, train_dice: {len(train_dice_data)}, val_data: {len(val_data)}')
       

        #-------------------------------------------------------------------------------#
        if args.step == "base_model":

            print("Step: Base model")
        
            model = train_model(train_data, train_dice_data, val_data, config, suffix, wandb_mode)

            
        elif args.step == "feature_segmentor":

            print("Step: Feature segmentor")
            train(train_data, train_dice_data, val_data, config, suffix, wandb_mode)

        elif args.step == "lora":
            print(f'step : {args.step}')
            if args.data == "cc359":
                train_data = cc359_refine(config, args.site)
                train_dice_data = cc359_3d_volume(config, args.site)
                val_data = cc359_3d_volume(config, args.site, train= False)

             
            train_model(train_data, train_dice_data, val_data, config, suffix, wandb_mode) #, add_lora=True)
             
        else:
            print("Step: Refinement")
      
            if args.data == "cc359":
          
                train_data = cc359_refine(config, args.site)
                train_dice_data = cc359_3d_volume(config, args.site)
                val_data = cc359_3d_volume(config, args.site, train= False)
                # embed()
             
            # if args.data == "mms":
            else:
                train_data = MMSDataset(config, args.site, train = True)
                train_dice_data = mms_3d_volume(config, args.site)
                val_data = mms_3d_volume(config, args.site, train= False)
                print(f'train: {len(train_data)}, train_dice: {len(train_dice_data)}, val_data: {len(val_data)}')
            # embed()

            # if args.step == "refine":
            #     print("refine")
            #     embed()
            print("udas refine")
            train_target(train_data, train_dice_data, val_data, config, 
                            suffix, wandb_mode,  add_lora=True)
            # else:
            # print("lora refine")
            # # # embed()
            # mix_labels(train_data, train_dice_data, val_data, config, 
            #             suffix, wandb_mode,   args.final_alpha, add_lora=True)
                            

   

    else:
        print('----------------------------------------------------------------------')
        print('                    Testing started ...')
        print('----------------------------------------------------------------------')
        print(f'dataset: {args.data}')
        # embed()
        if args.data == "cc359":
            test_data =   cc359_3d_volume(config, args.site, train= False)
            
        else:
            test_data =   mms_3d_volume(config, args.site, train= False)
            # embed()

        final_avg_dice, loss = test(test_data, config, suffix, wandb_mode)
        print(f"Final average dice score: {final_avg_dice}, Total loss: {loss}")
  


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDAS')

    # define arguments
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--suffix', type=str, required = True, help = "checkpoint suffix")
    parser.add_argument('--wandb_mode', type=str, required = True, help='wandb mode')
    parser.add_argument('--step', type= str, required= True, help="choose stage of doamina daptation")
    parser.add_argument('--test', type = None, help = "to turn test mode on")
    parser.add_argument('--seed', type = int, required = True, help = "random seed")
    # parser.add_argument('--alpha', type = int, help = "controls mixed ratio of psuedo labels")
    parser.add_argument('--final_alpha', type = float,  required = True, help = "final alpha value")
    parser.add_argument('--site', action='store', required = True, help = "can be int or str")
    parser.add_argument('--data', type = str, required = True, help = "specify the name of dataset")
    parser.add_argument('--checkpoint', type = str, help = "name of checkpoint")
    args = parser.parse_args()
    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    print('----------------------------------------------------------------------')
    print('Time: ' + now)
    print('----------------------------------------------------------------------')
    print('                    Now start ...')
    print('----------------------------------------------------------------------')

    main(args, now, args.suffix, args.wandb_mode)

    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')
