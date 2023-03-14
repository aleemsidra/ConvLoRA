# coding=utf-8
import os
import json
from easydict import EasyDict as edict

def process_config(jsonfile=None):
    try:
        if jsonfile is not None:
            with open(jsonfile, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    print("Config: ", config_args)
    print("\n")

    return config_args


def check_config_dict(config_dict):
    """
    check configuration
    :param config_dict: input config
    :return: 
    """
    if isinstance(config_dict["model_module_name"],str) is False:
        raise TypeError("model_module_name param input err...")
    if isinstance(config_dict["model_net_name"],str) is False:
        raise TypeError("model_net_name param input err...")
    if isinstance(config_dict["gpu_id"],str) is False:

        raise TypeError("gpu_id param input err...")
    if isinstance(config_dict["async_loading"],bool) is False:
        raise TypeError("async_loading param input err...")
    if isinstance(config_dict["is_tensorboard"],bool) is False:
        raise TypeError("is_tensorboard param input err...")
    if isinstance(config_dict["evaluate_before_train"],bool) is False:
        raise TypeError("evaluate_before_train param input err...")
    if isinstance(config_dict["shuffle"],bool) is False:
        raise TypeError("shuffle param input err...")
    if isinstance(config_dict["data_aug"],bool) is False:
        raise TypeError("data_aug param input err...")

    if isinstance(config_dict["num_epochs"],int) is False:
        raise TypeError("num_epochs param input err...")
    if isinstance(config_dict["img_height"],int) is False:
        raise TypeError("img_height param input err...")
    if isinstance(config_dict["img_width"],int) is False:
        raise TypeError("img_width param input err...")
    if isinstance(config_dict["num_channels"],int) is False:
        raise TypeError("num_channels param input err...")
    if isinstance(config_dict["num_classes"],int) is False:
        raise TypeError("num_classes param input err...")
    if isinstance(config_dict["batch_size"],int) is False:
        raise TypeError("batch_size param input err...")
    if isinstance(config_dict["dataloader_workers"],int) is False:
        raise TypeError("dataloader_workers param input err...")
    if isinstance(config_dict["learning_rate"],(int,float)) is False:
        raise TypeError("learning_rate param input err...")
    if isinstance(config_dict["learning_rate_decay"],(int,float)) is False:
        raise TypeError("learning_rate_decay param input err...")
    if isinstance(config_dict["learning_rate_decay_epoch"],int) is False:
        raise TypeError("learning_rate_decay_epoch param input err...")

    if isinstance(config_dict["train_mode"],str) is False:
        raise TypeError("train_mode param input err...")
    if isinstance(config_dict["file_label_separator"],str) is False:
        raise TypeError("file_label_separator param input err...")
    if isinstance(config_dict["pretrained_path"],str) is False:
        raise TypeError("pretrained_path param input err...")
    if isinstance(config_dict["pretrained_file"],str) is False:
        raise TypeError("pretrained_file param input err...")
    if isinstance(config_dict["save_path"],str) is False:
        raise TypeError("save_path param input err...")
    if isinstance(config_dict["save_name"],str) is False:
        raise TypeError("save_name param input err...")

    if not os.path.exists(os.path.join(config_dict["pretrained_path"], config_dict["pretrained_file"])):
        raise ValueError("cannot find pretrained_path or pretrained_file...")
    if not os.path.exists(config_dict["save_path"]):
        raise ValueError("cannot find save_path...")

    if isinstance(config_dict["train_data_root_dir"],str) is False:
        raise TypeError("train_data_root_dir param input err...")
    if isinstance(config_dict["val_data_root_dir"],str) is False:
        raise TypeError("val_data_root_dir param input err...")
    if isinstance(config_dict["train_data_file"],str) is False:
        raise TypeError("train_data_file param input err...")
    if isinstance(config_dict["val_data_file"],str) is False:
        raise TypeError("val_data_file param input err...")

    if not os.path.exists(config_dict["train_data_root_dir"]):
        raise ValueError("cannot find train_data_root_dir...")
    if not os.path.exists(config_dict["val_data_root_dir"]):
        raise ValueError("cannot find val_data_root_dir...")
    if not os.path.exists(config_dict["train_data_file"]):
        raise ValueError("cannot find train_data_file...")
    if not os.path.exists(config_dict["val_data_file"]):
        raise ValueError("cannot find val_data_file...")



#global_config = process_config('configs/config.json')

if __name__ == '__main__':
    config = global_config
    print(config['experiment_dir'])
    print('done')