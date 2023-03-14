# Refine UNet early features on the Calgary Campinas or M&Ms Dataset
# Author: Rasha Sheikh

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from dpipe.torch.functional import weighted_cross_entropy_with_logits

from .models import UNet2D
from .calgary_campinas_dataset import CalgaryCampinasDataset
from .evaluate import dice_score


def train_model(dataset_train, save_model_to, model, features_segmenter, save_log_to="log_exp", num_epochs=50, device=torch.device("cuda:0"), batch_size=20, initial_lr=0.001, stopping_thresh=0.005, level=0, n_channels_out=1):

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=10, drop_last=True)

    assert (level <= 1)

    features_segmenter.cuda(device)
    features_segmenter.eval()

    model.cuda(device)

    for p in model.parameters():
        p.requires_grad = False

    if level == 0:
        for p in model.init_path.parameters():
            p.requires_grad = True
    else:  # level 1
        for p in model.init_path.parameters():
            p.requires_grad = True
        for p in model.down1.parameters():
            p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)

    writer = SummaryWriter(log_dir=save_log_to)

    model.eval()
    train_dice_total = 0.0
    train_loss_total = 0.0
    num_steps = 0
    for i, batch in enumerate(train_loader):
        input_samples, _, _ = batch

        var_input = input_samples.cuda(device)

        stronger_preds = model(var_input)

        if level == 0:
            layer_activations = model.init_path(var_input)
            preds = features_segmenter(layer_activations)
        else:  # level = 1
            layer_activations_0 = model.init_path(var_input)
            layer_activations_1 = model.down1(layer_activations_0)
            logits_ = features_segmenter(layer_activations_1)
            preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')
            
        if n_channels_out == 1:
            stronger_preds_prob = torch.sigmoid(stronger_preds)
            loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
            dice = dice_score(torch.sigmoid(preds) > 0.5, stronger_preds_prob > 0.5)
        else:
            loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1))         
            dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)

        train_dice_total += dice.item()
        train_loss_total += loss.item()
        num_steps += 1

    train_dice_total_avg = train_dice_total / num_steps

    print('avg starting dice', train_dice_total_avg)
    train_dice_total_avg_old = train_dice_total_avg

    train_loss_total_avg = train_loss_total / num_steps
    print('avg starting loss', train_loss_total_avg)

    #    for epoch in tqdm(range(1, num_epochs+1)):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, _, _ = batch
            var_input = input_samples.cuda(device)

            stronger_preds = model(var_input)

            if level == 0:
                layer_activations = model.init_path(var_input)
                preds = features_segmenter(layer_activations)
            else:  # level = 1
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                logits_ = features_segmenter(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

            if n_channels_out == 1:
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
                dice = dice_score(torch.sigmoid(preds) > 0.5, stronger_preds_prob > 0.5)
            else:
                loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1))         
                dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)

            train_loss_total += loss.item()

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

        print('avg train loss', train_loss_total_avg)

        model.eval()
        train_dice_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples, _ = batch

            var_input = input_samples.cuda(device)

            stronger_preds = model(var_input)

            if level == 0:
                layer_activations = model.init_path(var_input)
                preds = features_segmenter(layer_activations)
            else:  # level = 1
                layer_activations_0 = model.init_path(var_input)
                layer_activations_1 = model.down1(layer_activations_0)
                logits_ = features_segmenter(layer_activations_1)
                preds = F.interpolate(logits_, scale_factor=2, mode='bilinear')

            if n_channels_out == 1:
                stronger_preds_prob = torch.sigmoid(stronger_preds)
                loss = weighted_cross_entropy_with_logits(preds, stronger_preds_prob)
                dice = dice_score(torch.sigmoid(preds) > 0.5, stronger_preds_prob > 0.5)
            else:
                loss = -torch.mean(F.log_softmax(preds, dim=1)*F.softmax(stronger_preds, dim=1))         
                dice = dice_score(torch.argmax(preds, dim=1), torch.argmax(stronger_preds, dim=1), n_outputs=n_channels_out)

            train_dice_total += dice.item()
            num_steps += 1
        train_dice_total_avg = train_dice_total / num_steps

        print('avg train dice', train_dice_total_avg)
        dice_diff = train_dice_total_avg - train_dice_total_avg_old
        print('avg train dice_diff', dice_diff)
        train_dice_total_avg_old = train_dice_total_avg

        writer.add_scalars('loss', {
            'train_loss': train_loss_total_avg}, epoch)
        writer.add_scalars('dice_pgt', {
            'dice_pgta': train_dice_total_avg}, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        if epoch % 2 == 0:
            torch.save(model.state_dict(), save_model_to)

        if abs(dice_diff) < stopping_thresh:
            print("----small dice difference...will stop refinement")
            break

    torch.save(model.state_dict(), save_model_to)

    return model
