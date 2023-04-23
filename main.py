import numpy as np
import os 

from requests import request
import torch
from torch.utils.data import DataLoader
import time
from calgary_campinas_dataset import CalgaryCampinasDataset , cc359_3d_volume
from utils.utils import process_config, check_config_dict
from evaluate import predict_sub, dice_score
from train_unet import train_model
from train_features_segmenter import train
import matplotlib.pyplot as plt
from test import test

import argparse
from IPython import embed



def main(args, now, suffix, wandb_mode):
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))


    # test = args.test
    # train_dataset, val_dataset  = get_data_loader(config)
    if not args.test :
        print('----------------------------------------------------------------------')
        print('                    Loading Data ...')
        print('----------------------------------------------------------------------')
      
        train_data = CalgaryCampinasDataset(config)
        train_dice_data = cc359_3d_volume(config)
        val_data = cc359_3d_volume(config, train= False)

        if args.step == "base_model":
            print("Base model")
            model = train_model(train_data, train_dice_data, val_data, config, suffix, wandb_mode)

        elif args.step == "feature_segmentor":

            print("Feature segmentor")
            train(train_data, train_dice_data, val_data, config, suffix, wandb_mode)

    else:
        print('----------------------------------------------------------------------')
        print('                    Testing started ...')
        print('----------------------------------------------------------------------')
        
        test_data =   cc359_3d_volume(config, train= False)
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




