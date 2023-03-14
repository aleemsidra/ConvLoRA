# Loader for the M&Ms dataset
# author: Rasha Sheikh

import numpy as np
import os
from collections import namedtuple
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

class MMSDataset(Dataset):
    def __init__(self, data_path, vendor="B", train=True, fold=-1, subj_index=[]):
       
        self.data_path = data_path
        self.fold = fold
        self.train = train
        self.subj_index = subj_index
        self.vendor = vendor
        self.one_hot_encoding = True
        self.n_classes = 4
        
        self.load_dataset_information()
        self.filter_data()
        self.load_files()
        
        
    def load_dataset_information(self):
        self.meta_info = {}
        file_path = os.path.join(self.data_path, '211006_M&Ms_Dataset_Information_-_opendataset.csv')
        with open(file_path) as f:
            file_content = f.readlines()
        header = file_content[0].strip().split(',')
        Meta = namedtuple('Meta', header[1:])
        for line in file_content[1:]:
            sample = line.strip().split(',')
            self.meta_info[sample[0]] = Meta(*sample[1:])
            
        
    def get_fold(self, files):
        kf = KFold(n_splits=3)
        folds = kf.split(files)
        k_i = 1
        for train_indices, test_indices in folds:
            if k_i == self.fold:
                if self.train:
                    indices = train_indices
                else:
                    indices = test_indices
                break
            k_i+=1
        return files[indices]
    
            
    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape
        b = (max_size[0] - current_size[0]) // 2
        a = max_size[0]-(b+current_size[0])
        d = (max_size[1] - current_size[1]) // 2
        c = max_size[1]-(d+current_size[1])
        return np.pad(data_array, ((b,a),(d,c),(0,0)), mode='edge') 
    
                
    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros((len(input_images),2), np.int)
        for i in range(len(input_images)):
            sizes[i,0] = input_images[i].shape[0]
            sizes[i,1] = input_images[i].shape[1]
        max_size = np.max(sizes, axis=0)
        max_size = self.validate_size(max_size)
        for i in range(len(input_images)):
            if sizes[i,0] != max_size[0] or sizes[i,1] != max_size[1]:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels
    
    
    def validate_size(self, size):
        H, W = size
        rem = H%8
        if rem != 0:
            new_H = H + 8-rem
        else:
            new_H = H
        rem = W%8
        if rem != 0:
            new_W = W + 8-rem
        else:
            new_W = W
        new_size = (new_H, new_W)
        
        return new_size
    
    
    def filter_data(self):
        if self.train:
            self.images_path = os.path.join(self.data_path, 'Training', 'Labeled')
        else:
            self.images_path = os.path.join(self.data_path, 'Testing')
        all_files = os.listdir(self.images_path)
        files = []
        for f in all_files:
            if self.meta_info[f].Vendor == self.vendor:
                files.append(f)
        files = np.array(sorted(files))
        if self.fold > 0:
            files = self.get_fold(files)
        if len(self.subj_index) > 0:
            files = files[self.subj_index]
        self.files = files
    
        
    def load_files(self):

        scaler = MinMaxScaler()
        images = []
        labels = []
        self.voxel_dim = []

        for i, f in enumerate(self.files):
            nib_file = nib.load(os.path.join(self.images_path, f, f+'_sa.nii.gz'))
            img = nib_file.get_fdata('unchanged', dtype=np.float32)
            ED = int(self.meta_info[f].ED)
            ES = int(self.meta_info[f].ES)
            img = np.stack([img[:,:,:,ED], img[:,:,:,ES]], axis=3)
            lbl_nib_file = nib.load(os.path.join(self.images_path, f, f+'_sa_gt.nii.gz'))
            lbl = lbl_nib_file.get_fdata('unchanged', dtype=np.float32)
            lbl = np.stack([lbl[:,:,:,ED], lbl[:,:,:,ES]], axis=3)
            transformed = scaler.fit_transform(np.reshape(img, (-1,1)))
            img = np.reshape(transformed, img.shape)
            img = np.reshape(img,(img.shape[0], img.shape[1], -1))
            images.append(img)
            lbl = np.reshape(lbl,(lbl.shape[0], lbl.shape[1], -1))
            labels.append(lbl)
            spacing = [nib_file.header.get_zooms()]*img.shape[0]
            self.voxel_dim.append(np.array(spacing))
                
        images, labels = self.unify_sizes(images, labels)
            
        self.voxel_dim = np.vstack(self.voxel_dim)
        self.data = np.expand_dims(np.moveaxis(np.concatenate(images, axis=-1),-1,0), axis=1)
        labels = np.moveaxis(np.concatenate(labels, axis=-1),-1,0)
        
        if self.one_hot_encoding:
            shape = labels.shape
            self.label = np.zeros((shape[0], self.n_classes, shape[1], shape[2]), dtype=np.int_)
            for c in range(self.n_classes):
                self.label[:,c,:,:] = labels[:,:,:]==c
        else:
            self.label = labels.astype(np.int_)
            self.label = np.expand_dims(self.label, axis=1)
                    
        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.voxel_dim = torch.from_numpy(self.voxel_dim)
        
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]
        
        return data, labels, voxel_dim
    
    