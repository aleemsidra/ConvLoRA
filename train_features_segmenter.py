# Train FeaturesSegmenter model on the Calgary Campinas or M&Ms Dataset
# Author: Rasha Sheikh

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits

from .models import FeaturesSegmenter
from .evaluate import dice_score


def train_model(dataset_train, dataset_val, save_model_to, unet_model, save_log_to="log_exp", num_epochs=50,
                device=torch.device("cuda:0"), batch_size=20, initial_lr=0.001, level=0, n_channels_out=1):

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=10, drop_last=True)

    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=False, num_workers=10, drop_last=False)

    assert (level <= 1)

    in_channels = 16 * (2 ** level)
    model = FeaturesSegmenter(in_channels=in_channels, out_channels=n_channels_out)
    model.cuda(device)

    unet_model.cuda(device)
    unet_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    
    CE_loss = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=save_log_to)

    #    for epoch in tqdm(range(1, num_epochs+1)):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples, _ = batch

            var_input = input_samples.cuda(device)
            var_gt = gt_samples.cuda(device, non_blocking=True)

            if level == 0:
                layer_activations = unet_model.init_path(var_input)
                preds = model(layer_activations)
            else:  # level = 1
                layer_activations_0 = unet_model.init_path(var_input)
                layer_activations_1 = unet_model.down1(layer_activations_0)
                logits_ = model(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')
            
            if n_channels_out == 1:
                loss = weighted_cross_entropy_with_logits(preds, var_gt)
                dice = dice_score(torch.sigmoid(preds) > 0.5, var_gt)
            else:
                loss = CE_loss(preds, torch.argmax(var_gt, dim=1))          
                dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(var_gt, dim=1), n_outputs=n_channels_out)

            train_loss_total += loss.item()
            train_dice_total += dice.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            if epoch % 30 == 0 or epoch % num_epochs == 0:
                grid_img = vutils.make_grid(input_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu()[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Train Ground Truth', grid_img, epoch)

        train_loss_total_avg = train_loss_total / num_steps
        train_dice_total_avg = train_dice_total / num_steps

        print('avg train loss', train_loss_total_avg)
        print('avg train dice', train_dice_total_avg)

        model.eval()

        val_loss_total = 0.0
        val_dice_total = 0.0

        num_steps = 0

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples, _ = batch

            with torch.no_grad():
                var_input = input_samples.cuda(device)
                var_gt = gt_samples.cuda(device, non_blocking=True)

                if level == 0:
                    layer_activations = unet_model.init_path(var_input)
                    preds = model(layer_activations)
                else:  # level = 1
                    layer_activations_0 = unet_model.init_path(var_input)
                    layer_activations_1 = unet_model.down1(layer_activations_0)
                    logits_ = model(layer_activations_1)
                    preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

                if n_channels_out == 1:
                    loss = weighted_cross_entropy_with_logits(preds, var_gt)
                    dice = dice_score(torch.sigmoid(preds) > 0.5, var_gt)
                else:
                    loss = CE_loss(preds, torch.argmax(var_gt, dim=1))          
                    dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(var_gt, dim=1), n_outputs=n_channels_out)

                val_loss_total += loss.item()
                val_dice_total += dice.item()

            num_steps += 1

            if epoch % 30 == 0 or epoch % num_epochs == 0:
                grid_img = vutils.make_grid(input_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu()[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples[:4],
                                            normalize=False,
                                            scale_each=False)
                writer.add_image('Val Ground Truth', grid_img, epoch)

        val_loss_total_avg = val_loss_total / num_steps
        val_dice_total_avg = val_dice_total / num_steps

        print('avg val loss', val_loss_total_avg)
        print('avg val dice', val_dice_total_avg)

        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg,
            'val_loss': val_loss_total_avg
        }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        if epoch % 2 == 0:
            torch.save(model.state_dict(), save_model_to)

    torch.save(model.state_dict(), save_model_to)

    return model
