import numpy as np
import random
import os 

# from requests import request
import torch
from torch.utils.data import DataLoader
import time

from calgary_campinas_dataset import CalgaryCampinasDataset , cc359_3d_volume, cc359_refine
from mms_dataset import MMSDataset, mms_3d_volume
from utils.utils import process_config

from train_unet import train_model
from train_features_segmenter import early_feature_segmentor
from adaptation import target_adaptation

import matplotlib.pyplot as plt
from test import test

import argparse
from IPython import embed
import torch


def main(args, now, suffix, wandb_mode):

    # Initializing seeds and preparing GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  
    torch.manual_seed(args.seed)  
    random.seed(args.seed)  
    np.random.seed(args.seed)
   
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))

    if args.site.isdecimal():
        args.site = int(args.site)

    if not args.test :

        print('----------------------------------------------------------------------')
        print('                    Loading Data ...')
        print('----------------------------------------------------------------------')
        
        print(f"step: {args.step}, data: {args.data}")
  
        if args.step == "base_model" or args.step == "feature_segmentor":
            print(f'dataset: {args.data}')
            train_data = CalgaryCampinasDataset(config,  args.site)
            train_dice_data = cc359_3d_volume(config, args.site)
            val_data = cc359_3d_volume(config, args.site,train= False)
            print(f'train: {len(train_data)}, train_dice: {len(train_dice_data)}, val_data: {len(val_data)}')
         
            
        #-------Respective training based on step----------#
        if args.step == "base_model":
            # Train Base Model
            print(f"Step: {args.step}")
            train_model(train_data, train_dice_data, val_data, config, suffix, wandb_mode)

        elif args.step == "feature_segmentor":
            # Train ESH
            print(f"Step: {args.step}")
            early_feature_segmentor(train_data, train_dice_data, val_data, config, suffix, wandb_mode)
  
        else:
             # Target Domain Adaptation
            print(f"Step: {args.step}")
            train_data = cc359_refine(config, args.site)
            train_dice_data = cc359_3d_volume(config, args.site)
            val_data = cc359_3d_volume(config, args.site, train= False)
            print(f'data loaded:  {args.data} ')
            print(f'train: {len(train_data)}, train_dice: {len(train_dice_data)}, val_data: {len(val_data)}')

            target_adaptation(train_data, train_dice_data, val_data, args.adapt, config, 
                          suffix, wandb_mode)
                                
    else:
        print('----------------------------------------------------------------------')
        print('                    Testing started ...')
        print('----------------------------------------------------------------------')
        print(f'dataset: {args.data}')

        if args.data == "cc359": 
            test_data =   cc359_3d_volume(config, args.site, train= False)
            
        else:
            test_data =   mms_3d_volume(config, args.site, train= False)
            
        final_avg_dice, loss = test(test_data, args.adapt, config, suffix, wandb_mode)
        print(f"Final average dice score: {final_avg_dice}, Total loss: {loss}")
  


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDAS')

    # define arguments
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--suffix', type=str, required = True, help = "checkpoint suffix")
    parser.add_argument('--wandb_mode', type=str, required = True, help='wandb mode')
    parser.add_argument('--step', type= str, required= True, help="choose stage of doamin daptation: base_model/feature_segmentor/adapt")
    parser.add_argument('--test', type = None, help = "to turn test mode on")
    parser.add_argument('--seed', type = int, required = True, help = "random seed")
    parser.add_argument('--adapt', type = str,  required = False, help = "specfiy adaptation method")
    parser.add_argument('--site', action='store', required = True, help = "can be int or str")
    parser.add_argument('--data', type = str, required = True, help = "specify the name of dataset")
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
