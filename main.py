import numpy as np
import os 

from requests import request
import torch
from torch.utils.data import DataLoader
import time
from calgary_campinas_dataset import CalgaryCampinasDataset #, get_data_loader
from utils.utils import process_config, check_config_dict
from train_unet import train_model
import argparse




def main(args, now, suffix, wandb_mode):
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))
    # print("config", config)
    # print(config.site, config.data_path)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # dataset = CalgaryCampinasDataset(config)

    print('                    Loading Data ...')
    print('----------------------------------------------------------------------')

    # train_dataset, val_dataset  = get_data_loader(config)

    train_data = CalgaryCampinasDataset(config)
    val_data   = CalgaryCampinasDataset(config, train = False, subj_index= list(range(0, 10, 1)))

    model = train_model(train_data,val_data, config, suffix, wandb_mode)
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDAS')

    # define arguments
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--suffix', type=str, required = True, help = "path to config file")
    parser.add_argument('--wandb_mode', type=str, help='wandb mode')


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




