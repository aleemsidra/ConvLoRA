# Functions to support the evaluation of segmentations
# Author: Rasha Sheikh

import numpy as np
import torch
from torch.utils.data import DataLoader
from save_model import load_model
import surface_distance.metrics as surf_dc
from models import UNet2D
from IPython import embed
from calgary_campinas_dataset import CalgaryCampinasDataset


def predict_subjects_segmentation(subj_ids, model, data_path, device, site=4, train=True, fold=2, batch_size=16):
    input_mul_subjects = []
    gt_mul_subjects = []
    preds_mul_subjects = []
    voxel_dim_mul_subjects = []

    step_size = len(subj_ids) // min(10, len(subj_ids))
    assert (step_size > 0)
    end_ = step_size * min(10, len(subj_ids))

    for i in range(0, end_, step_size):
        all_input, all_gt, all_preds, all_voxel_dim = predict_sub(model, data_path, site, train, fold, subj_ids[i],
                                                                  device, batch_size)

        input_mul_subjects.append(all_input)
        gt_mul_subjects.append(all_gt)
        preds_mul_subjects.append(all_preds)
        voxel_dim_mul_subjects.append(all_voxel_dim)

    input_mul_subjects, gt_mul_subjects, preds_mul_subjects = unify_shapes(
        [input_mul_subjects, gt_mul_subjects, preds_mul_subjects])

    input_mul_subjects = np.concatenate(input_mul_subjects)
    gt_mul_subjects = np.concatenate(gt_mul_subjects)
    preds_mul_subjects = np.concatenate(preds_mul_subjects)

    return [input_mul_subjects, gt_mul_subjects, preds_mul_subjects], voxel_dim_mul_subjects


# def predict_sub(model, data_path, site, train, fold, subj_index, device, batch_size=16):


#     dataset = CalgaryCampinasDataset(data_path, site=site, train=train, fold=fold, subj_index=[subj_index])



#     # test_data = CalgaryCampinasDataset(data_path, site, train = False, subj_index= list(range(10, 20, 1)))  use this for test

#     loader = DataLoader(dataset, batch_size=batch_size,
#                         shuffle=False, num_workers=10, drop_last=False)

#     model.eval()

#     all_pred = []
#     all_input = []
#     all_gt = []
#     all_voxel_dim = []

#     for b, batch in enumerate(loader):
#         input_samples, gt_samples, voxel_dim = batch

#         with torch.no_grad():
#             var_input = input_samples.cuda(device)
# #            preds = torch.mean(model(var_input)[1:], dim=0, keepdim=True)
#             preds = model(var_input)

#             all_pred.append(preds.cpu().numpy())
#             all_input.append(var_input.cpu().numpy())
#             all_gt.append(gt_samples)
#             all_voxel_dim.append(voxel_dim.numpy())

#     all_pred = np.concatenate(all_pred, axis=0)
#     all_input = np.concatenate(all_input, axis=0)
#     all_gt = np.concatenate(all_gt, axis=0)
#     all_voxel_dim = np.concatenate(all_voxel_dim, axis=0)

#     return all_input, all_gt, all_pred, all_voxel_dim


def predict_sub(config):

    # data_path = config.data_path
    # site = config.site
    # fold = config.fold


    # print(config)
    batch_size = config.batch_size
    checkpoint = config.checkpoint
  
    device=torch.device("cuda:0")

    # dataset = CalgaryCampinasDataset(data_path, site, fold, train=False, subj_index=[])

    dataset = CalgaryCampinasDataset(config, train=False ) #, subj_index=[])

    # test_data = CalgaryCampinasDataset(data_path, site, train = False, subj_index= list(range(10, 20, 1)))  use this for test

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=False)

    model = load_model(config)
    model.cuda(device)
    
    model.eval()

    all_pred = []
    all_input = []
    all_gt = []
    all_voxel_dim = []

    for b, batch in enumerate(loader):
        input_samples, gt_samples, voxel_dim = batch

        with torch.no_grad():
            var_input = input_samples.cuda(device)
#            preds = torch.mean(model(var_input)[1:], dim=0, keepdim=True)
            preds = model(var_input)

            # all_pred.append(preds.cpu().numpy())
            # all_input.append(var_input.cpu().numpy())
            # all_gt.append(gt_samples)
            # all_voxel_dim.append(voxel_dim.numpy())

            preds = torch.sigmoid(preds) > 0.5
            
            all_pred.append(preds.cpu().numpy())
            all_input.append(var_input.cpu().numpy())
            all_gt.append(gt_samples)
            all_voxel_dim.append(voxel_dim.numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_input = np.concatenate(all_input, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_voxel_dim = np.concatenate(all_voxel_dim, axis=0)

    return all_input, all_gt, all_pred, all_voxel_dim


def evaluate_preds_surface_dice(ground_truth, final_output, voxel_dim):
    tolerance = 1.0
    all_surface_dice = []

    s = 0
    for subj in range(len(voxel_dim)):
        vol_size = len(voxel_dim[subj])
        spacing = voxel_dim[subj][0]
        embed()
        # surface_distances = surface_distance.metrics.compute_surface_distances(sigmoid(final_output[s:s + vol_size]) > 0.5,
        #                                                                        ground_truth[s:s + vol_size] > 0,
        #                                                                        spacing)
        
        surface_distances = surface_distance.metrics.compute_surface_distances(final_output[s:s + vol_size] ,
                                                                               ground_truth[s:s + vol_size] ,
                                                                               spacing)
        surface_dice = surface_distance.metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance)

        print("surface of one image", surface_dice)
        # embed()
        all_surface_dice.append(surface_dice)

        s += vol_size

    avg_surf_dice = np.average(all_surface_dice)

    print('avg_surf_dice', avg_surf_dice)
    # embed()
    return avg_surf_dice




def unify_shapes(data):
    input_images = data[0]
    sizes = np.zeros(len(input_images), int)
    for i in range(len(input_images)):
        sizes[i] = input_images[i].shape[-1]
    max_size = np.max(sizes)
    for i in range(len(input_images)):
        if sizes[i] != max_size:
            for j in range(len(data)):
                data[j][i] = pad_data(data[j][i], max_size)
    return data


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp([-x for x in x]))
 


# def dice_score(input_, target, n_outputs, print_=False, average=True):
def dice_score(input_, target, n_outputs, print_=False, average=True):
## for M&M
    eps = 0.0001
    
    iflat = input_.reshape(-1)
    tflat = target.reshape(-1)
    
    if n_outputs == 1:
        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        if print_:
            print(dice)
    else:
        dice = torch.zeros(n_outputs - 1)
        # dice = torch.zeros(n_outputs)

        for c in range(1, n_outputs):
            iflat_ = iflat == c
            tflat_ = tflat == c
            
            intersection = torch.sum(iflat_ * tflat_)
            union = torch.sum(iflat_) + torch.sum(tflat_)
            d = (2.0 * intersection + eps) / (union + eps)
            
            if print_:
                print(c, d.item())
            dice[c - 1] += d
            # dice[c] += d
        print("dice,", dice)
        # embed()
        if average:
            avg_dice = torch.sum(dice) / (n_outputs - 1)
         
    if n_outputs ==1:
        return  dice
    else:
        return avg_dice, dice

def pad_data(data_array, max_size):
    current_size = data_array.shape[-1]
    b = (max_size - current_size) // 2
    a = max_size-(b+current_size)
    return np.pad(data_array, ((0,0),(0,0),(b,a),(b,a)), mode='edge')




def sdice(a, b, spacing,tolerance= 1):

    # update this code to loop over segemnted volumeS (LINE 170 in train_unet)
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)



