# Loader for the Calgary Campinas dataset


import numpy as np
import pandas as pd
import tqdm
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import imageio
from IPython import embed

class CalgaryCampinasDataset(Dataset):
    # load dataset in 2D
    def __init__(self, config,  site, train=True,  rotate=True, scale=True ):
        self.rotate = rotate
        self.scale = scale
        self.fold = config.fold
        self.train = train
        self.site = site
        self.data_path = config.data_path
        self.source = config.source

        if self.site == 1:
            self.folder = 'GE_15'
        elif self.site == 2:
            self.folder = 'GE_3'
        elif self.site == 3:
            self.folder = 'Philips_15'
        elif self.site == 4:
            self.folder = 'Philips_3'
        elif self.site == 5:
            self.folder = 'Siemens_15'
        elif self.site == 6:
            self.folder = 'Siemens_3'
        else:
            self.folder = 'GE_3'

        self.load_files(self.data_path)

    def pad_image(self, img):
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode='edge')
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode='edge')
        else:
            return img

    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode='edge')

    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros(len(input_images), int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)
        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels

    def load_files(self, data_path):
      
        
        self.sagittal = True

        scaler = None
        if self.scale:
            scaler = MinMaxScaler()
        images = []
        labels = []
        self.voxel_dim = [] 
     
        images_path = os.path.join(data_path, 'Original', self.folder)
        print("images_path", images_path )

        if self.source and self.train:

            self.images_path = os.path.join(data_path, 'Original', self.folder, "train")
      
            print("train_path ", self.images_path )
        

        elif self.source and not self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val")
            print("val_path ",self.images_path)


        else:
            print(self.source, self.train)
            self.images_path = os.path.join(data_path, 'Original', self.folder)
            print("image_path ", self.images_path)
        
  
        files = np.array(sorted(os.listdir(self.images_path)))
        for i, f in enumerate(files):
  
            nib_file = nib.load(os.path.join(self.images_path, f))
            img = nib_file.get_fdata('unchanged', dtype=np.float32) #loadibg metadata
            slice_range =(25,175) # selected after manual inspection

            img = img[slice_range[0]:slice_range[1]+1, :, :]

            lbl = nib.load(os.path.join(data_path, 'Silver-standard', self.folder, f[:-7] + '_ss.nii.gz')).get_fdata(
                'unchanged', dtype=np.float32)
            lbl = lbl[slice_range[0]:slice_range[1]+1, :, :]

            if self.scale:
                transformed = scaler.fit_transform(np.reshape(img, (-1, 1)))
                img = np.reshape(transformed, img.shape)
            if not self.sagittal:
                img = np.moveaxis(img, -1, 0)
            if self.rotate:
                img = np.rot90(img, axes=(1, 2))
            if img.shape[1] != img.shape[2]:
                img = self.pad_image(img)
            images.append(img)

            if not self.sagittal:
                lbl = np.moveaxis(lbl, -1, 0)
            if self.rotate:
                lbl = np.rot90(lbl, axes=(1, 2))
            if lbl.shape[1] != lbl.shape[2]:
                lbl = self.pad_image(lbl)
            labels.append(lbl)

            spacing = [nib_file.header.get_zooms()] * img.shape[0]
            self.voxel_dim.append(np.array(spacing))  
   
        images, labels = self.unify_sizes(images, labels)

        self.data = np.expand_dims(np.vstack(images), axis=1)
        self.label = np.expand_dims(np.vstack(labels), axis=1)
        self.voxel_dim = np.vstack(self.voxel_dim)

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





# Loader for the Calgary Campinas dataset


class cc359_refine(Dataset):
    # def __init__(self, data_path, site=2, train=True, fold=-1, rotate=True, scale=True, subj_index=[]):
    def __init__(self, config, site, train=True,  rotate=True, scale=True, subj_index=[]):
        self.rotate = rotate
        self.scale = scale
        self.fold = config.fold
        self.train = train
        self.subj_index = subj_index
        self.stage = config.stage
        # self.site = config.site
        self.site = site
        self.data_path = config.data_path
        self.source = config.source


        if self.site == 1:
            self.folder = 'GE_15'
            self.range = (60,195)
        elif self.site == 2:
            self.folder = 'GE_3'
            self.range = (25,175)
        elif self.site == 3:
            self.folder = 'Philips_15'
            self.range = (10,150)
        elif self.site == 4:
            self.folder = 'Philips_3'
            self.range = (20,155)
        elif self.site == 5:
            self.folder = 'Siemens_15'
            self.range = (25,165)
        elif self.site == 6:
            self.folder = 'Siemens_3'
            self.range = (60,165)
        else:
            self.folder = 'GE_3'

        self.load_files(self.data_path)

     

    def pad_image(self, img):
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode='edge')
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode='edge')
        else:
            return img

    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode='edge')

    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros(len(input_images), int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)
        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels

    def load_files(self, data_path):
      
        
        self.sagittal = True

        scaler = None
        if self.scale:
            scaler = MinMaxScaler()
        images = []
        labels = []
        self.voxel_dim = [] 
     


        if self.stage == "refine" and self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "train.csv")
            print("train_path ", self.images_path )

        files =pd.read_csv(self.images_path, header=None).values.ravel().tolist()
  
        for i, f in enumerate(files):

            nib_file = nib.load(f)
            img = nib_file.get_fdata('unchanged', dtype=np.float32) #loadibg metadata
            img = img[self.range[0]:self.range[1]+1, :, :]
           
            lbl = nib.load(os.path.join(data_path, 'Silver-standard', self.folder, os.path.basename(f).split(".")[0] + '_ss.nii.gz')).get_fdata(
                'unchanged', dtype=np.float32)
            lbl = lbl[self.range[0]:self.range[1]+1, :, :]

            if self.scale:
                transformed = scaler.fit_transform(np.reshape(img, (-1, 1)))
                img = np.reshape(transformed, img.shape)
            if not self.sagittal:
                img = np.moveaxis(img, -1, 0)
            if self.rotate:
                img = np.rot90(img, axes=(1, 2))
            if img.shape[1] != img.shape[2]:
                img = self.pad_image(img)
            images.append(img)

          

            if not self.sagittal:
                lbl = np.moveaxis(lbl, -1, 0)
            if self.rotate:
                lbl = np.rot90(lbl, axes=(1, 2))
            if lbl.shape[1] != lbl.shape[2]:
                lbl = self.pad_image(lbl)
            labels.append(lbl)

            spacing = [nib_file.header.get_zooms()] * img.shape[0]
      
            self.voxel_dim.append(np.array(spacing))  

   
        images, labels = self.unify_sizes(images, labels)


        self.data = np.expand_dims(np.vstack(images), axis=1)
        self.label = np.expand_dims(np.vstack(labels), axis=1)
        self.voxel_dim = np.vstack(self.voxel_dim)

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




class cc359_3d_volume(Dataset):
    # loads data as 3D volume
    def __init__(self, config, site, train = True, rotate=True, scale=True ):
        self.rotate = rotate
        self.scale = scale
        self.fold = config.fold
        self.train = train
  
        self.site = site
        self.data_path = config.data_path
        self.source = config.source
 
        if self.site == 1:
            self.folder = 'GE_15'
            self.range = (60,195)
        elif self.site == 2:
            self.folder = 'GE_3'
            self.range = (25,175)
        elif self.site == 3:
            self.folder = 'Philips_15'
            self.range = (10,150)
        elif self.site == 4:
            self.folder = 'Philips_3'
            self.range = (20,155)
        elif self.site == 5:
            self.folder = 'Siemens_15'
            self.range = (25,165)
        elif self.site == 6:
            self.folder = 'Siemens_3'
            self.range = (60,165)
        else:
            self.folder = 'GE_3'

        print(f"folder: {self.folder}, slice_range: {self.range}")
        self.load_files(self.data_path)


    def load_files(self, data_path):
        
        self.sagittal = True
        if  self.source == "True" and self.train:
            
            self.images_path = os.path.join(data_path, 'Original', self.folder, "train.csv")
            print("train_path ", self.images_path )

        elif self.source == "True" and not self.train:
   
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val.csv")
            print("val_path ",self.images_path)


        else:
           
            print(self.source, self.train)
            self.images_path = os.path.join(data_path, 'Original', self.folder, "test.csv")   # replace it for rest of domains
            print("test_path ", self.images_path)
        
        self.volume_files = pd.read_csv(self.images_path, header=None).values.ravel().tolist()

    def img_transform(self, img):
          
        self.sagittal = True
        scaler = None
        if self.scale:
            scaler = MinMaxScaler()
        
        if self.scale:
            transformed = scaler.fit_transform(np.reshape(img, (-1, 1)))
            img = np.reshape(transformed, img.shape)
            
        if not self.sagittal:
            img = np.moveaxis(img, -1, 0)
            
        if self.rotate:
            img = np.rot90(img, axes=(1, 2))
        if img.shape[1] != img.shape[2]:
            img = self.pad_image(img)

        return img
    
    
    def pad_image(self, img):
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode='edge')
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode='edge')
        else:
            return img

    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode='edge')

    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros(len(input_images), int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)
        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels


    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        
        img = nib.load(self.volume_files[idx]).get_fdata('unchanged', dtype=np.float32)       
        nib_file = nib.load(self.volume_files[idx])
        slice_range = self.range
        spacing = [nib_file.header.get_zooms()] * nib_file.shape[0]
        self.voxel_dim= np.array(spacing)

        img = img[slice_range[0]:slice_range[1]+1, :, :]
        img = self.img_transform(img)  
        
        lbl = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, self.volume_files[idx][:-7].split("/")[-1] + '_ss.nii.gz')).get_fdata('unchanged', dtype=np.float32)
        lbl = self.img_transform(lbl)  
        lbl = lbl[slice_range[0]:slice_range[1]+1, :, :]
        
        images, labels = self.unify_sizes(img, lbl)
        data = torch.from_numpy(np.expand_dims(images.copy(), axis=1))
        label = torch.from_numpy(np.expand_dims(labels.copy(), axis=1))
        voxel_dim = torch.from_numpy(self.voxel_dim)
     
        return data, label, voxel_dim

