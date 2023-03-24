import numpy as np
import os 

from requests import request
import torch
from torch.utils.data import DataLoader
import time
from calgary_campinas_dataset import CalgaryCampinasDataset , cc359_volume#, get_data_loader
from utils.utils import process_config, check_config_dict
from evaluate import predict_sub, dice_score
from train_unet import train_model
import matplotlib.pyplot as plt

import argparse
from IPython import embed



def main(args, now, suffix, wandb_mode):
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))


    print('                    Loading Data ...')
    print('----------------------------------------------------------------------')
    # test = args.test
    # train_dataset, val_dataset  = get_data_loader(config)
    if not args.test :
        
        print('                    Training started ...')
        print('----------------------------------------------------------------------')
      
        train_data = CalgaryCampinasDataset(config)
        # train_dice = cc359_volume(config)

        val_data   = cc359_volume(config, train = False)



        # print("train_data",train_data[0][0].shape)

        # print("val_data",val_data[0][0].shape)

      

        # train_loader = DataLoader(train_data, batch_size=config.batch_size,
        #                       shuffle=True, num_workers=10, drop_last=True)
        
        val_loader = DataLoader(val_data, batch_size=config.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)


        # Display image and label.
        train_features, train_labels, voxel = next(iter(val_loader))
        
        # train_features, train_labels = next(iter(val_loader))

        # print(f"Feature batch shape: {train_features.size()}")
        # print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0].squeeze().permute(1,2,0)[:,:,127] #getting slice from imag
        # plt.imshow(img, cmap="gray")
        # plt.show()
      
     

        model = train_model(train_data, val_data, config, suffix, wandb_mode)
   
      
        # train_data = cc359_volume(config)
        # print("type", type(train_data)) # it should give 224,1,256,256 , but giving other wise
     
        # print("shape", train_data[1][0].data.shape, train_data[1][1].data.shape)
        asd
        # print("shape",train_data[0][1].shape, train_data[0][2]) # voxel dim should be 224,3 (for one image)
    
        # print("len", len(train_data))

        # asd

        train_loader = DataLoader(train_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=10, drop_last=True)
        
        # Display image and label.
        train_features, train_labels = next(iter(train_loader))
        embed()
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

        asd

        # print(train_features[0].shape)

        # img = train_features[0].squeeze()
    
        # label = train_labels[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
      
        # asd
      
        model = train_model(train_data,val_data, config, suffix, wandb_mode)
    else:
        
        print('                    Testing started ...')
        print('----------------------------------------------------------------------')
        # asd
        all_input, all_gt, all_pred, all_voxel_dim = predict_sub(config)
        dice = dice_score(all_pred > 0.5, all_gt)
        print("Test Dice Score", dice)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDAS')

    # define arguments
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--suffix', type=str, required = True, help = "checkpoint suffix")
    parser.add_argument('--wandb_mode', type=str, required = True, help='wandb mode')
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




