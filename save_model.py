import os
import json
import torch
from models import UNet2D
def save_model( model, config, suffix, folder_time):
        """
        implement the logic of saving model
        """
        print("Saving model...")
        save_path = config.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # print("path :" , save_path)
     


        try:
            if not os.path.exists(os.path.join (save_path, config.model_net_name)):
                save_dir = os.path.join(save_path, config.model_net_name + "_"+ suffix +"_" + folder_time)
                os.makedirs(save_dir)
        except:
             pass

        #save_name = os.path.join(save_path,self.config['save_name'])
        
        save_name = os.path.join(save_dir, config.save_name)

        torch.save(model.state_dict(), save_name)


def load_model(config):
    print("loading checkpoint")
    checkpoint = config.checkpoint
    n_channels_out = config.n_channels_out
    model = UNet2D(n_chans_in=1, n_chans_out=n_channels_out, n_filters_init=16)
    model.load_state_dict(torch.load(checkpoint))

    return model