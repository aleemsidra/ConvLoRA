import os
import json
# from __future__ import print_function
import os, sys
import numpy as np
import logging
from datetime import datetime
import time
from utils import utils
# from utils import logger_summarizer
import json
# from utils.utils import pr
# ocess_config

from torch.utils.tensorboard import SummaryWriter





def save_config(config, suffix, folder_time):
    

    print(config.save_log_to)
    log_dir = config.save_log_to
    print("log_dir", log_dir)
    
    try:
        if not os.path.exists(os.path.join(log_dir, "logs")):
                os.makedirs(log_dir)
    except: 
        pass
                
    try:
        if not os.path.exists(os.path.join (log_dir, config.model_net_name)):
            model_dir = os.path.join(log_dir, config.model_net_name + "_" +suffix+"_" + folder_time)
            os.makedirs(model_dir)
    except: 
        pass

    # saving config file
    with open(os.path.join(model_dir, 'config.json'), 'w') as fp:  
        json.dump(config, fp)
   

# def write_info_to_logger(writer, variable_dict):
#         """
#         print
#         :param variable_dict: 
#         :return: 
#         """
#         # log_info = {}
#         if variable_dict is not None:
#             for tag, value in variable_dict.items():
#                 # log_info[tag] = value
#                 writer.add_scalar(tag, value)


#         _info = 'epoch: %d, lr: %f, eval_train: %f, train_auc: %f, eval_validate: %f,  val_auc: %f, train_avg_loss: %f, validate_avg_loss: %f, gpu_index: %s, net: %s' % (
#         log_info['epoch'],log_info['lr'], log_info['train_acc'], log_info['train_auc'], log_info['validate_acc'], 
#         log_info['val_auc'], log_info['train_avg_loss'], log_info['validate_avg_loss'],
#         log_info['gpus_index'], log_info['net_name'])

#         self.log_writer.info(_info)
        # sys.stdout.flush()


# def write():
#         """
#         log writing
#         :return: 
#         """
#         _info = 'epoch: %d, lr: %f, eval_train: %f, train_auc: %f, eval_validate: %f,  val_auc: %f, train_avg_loss: %f, validate_avg_loss: %f, gpu_index: %s, net: %s' % (
#         log_info['epoch'],log_info['lr'], log_info['train_acc'], log_info['train_auc'], log_info['validate_acc'], 
#         log_info['val_auc'], log_info['train_avg_loss'], log_info['validate_avg_loss'],
#         log_info['gpus_index'], log_info['net_name'])

#         self.log_writer.info(_info)
#         sys.stdout.flush()