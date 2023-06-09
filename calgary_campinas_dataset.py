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
    # def __init__(self, data_path, site=2, train=True, fold=-1, rotate=True, scale=True, subj_index=[]):
    def __init__(self, config, train=True,  rotate=True, scale=True, subj_index=[]):
        self.rotate = rotate
        self.scale = scale
        self.fold = config.fold
        self.train = train
        self.subj_index = subj_index
        # print( "self.subj_index",  self.subj_index)
        # asd
        self.site = config.site
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

    def get_fold(self, files):
        kf = KFold(n_splits=3)
        
        folds = kf.split(files)
        # print("folds", folds)
        # asd
        k_i = 1
        for train_indices, test_indices in folds:
            
            if k_i == self.fold:
                if self.train:
                    # print("train_indices", train_indices )
                    indices = train_indices
                else:
                    # print("test_indices", test_indices)
                    indices = test_indices
                break
            k_i += 1
           
        # print("indices", indices, "\n", len(indices))
     
        return files[indices] 

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
        sizes = np.zeros(len(input_images), np.int)
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
     
        # images_path = os.path.join(data_path, 'Original', self.folder)
        # print("images_path", images_path )

        if self.source and self.train:

            # self.images_path = os.path.join(data_path, 'Original', self.folder, "train")
            self.images_path = os.path.join(data_path, 'Original', self.folder, "train.csv")
            print("train_path ", self.images_path )
        

        elif self.source and not self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val")
            print("val_path ",self.images_path)


        else:
            print(self.source, self.train)
            self.images_path = os.path.join(data_path, 'Original', self.folder)
            print("image_path ", self.images_path)
        
        files = np.array(sorted(os.listdir(self.images_path)))
        # print("length ", len(files))

        for i, f in enumerate(files):
            # asd
            # print("file_name", f)
            nib_file = nib.load(os.path.join(self.images_path, f))
            # print("nib_file_dim", nib_file.shape)
            img = nib_file.get_fdata('unchanged', dtype=np.float32) #loadibg metadata
            # print("image", img) # image loaded in np here
            # print("img tytpe", type(img))
           
            # print("img_num", i, "image_name:", f, "img", img.shape)
            

            # print("ground_truth_name:", os.path.join(data_path, 'Silver-standard', self.folder, f[:-7] + '_ss.nii.gz'))
            lbl = nib.load(os.path.join(data_path, 'Silver-standard', self.folder, f[:-7] + '_ss.nii.gz')).get_fdata(
                'unchanged', dtype=np.float32)
            # print("label shape", lbl.shape)
            # print("gt_name:", os.path.join(data_path, 'Silver-standard', self.folder, f[:-7] + '_ss.nii.gz'))
        
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

            

            # print("appended images", "len", len(images), "shape at first index",images[0].shape)
            # print(images)
          

            if not self.sagittal:
                lbl = np.moveaxis(lbl, -1, 0)
            if self.rotate:
                lbl = np.rot90(lbl, axes=(1, 2))
            if lbl.shape[1] != lbl.shape[2]:
                lbl = self.pad_image(lbl)
            labels.append(lbl)

            # print("labels appended ", "len", len(labels), "shape at first index", labels[0].shape)


            # print("img.shape[0]", img.shape[0])

            spacing = [nib_file.header.get_zooms()] * img.shape[0]
            # print("spacing", spacing)
            # asd
            self.voxel_dim.append(np.array(spacing))  

            # print("voxel appended ", len(self.voxel_dim), "shape at first index", self.voxel_dim[0].shape,)
        
            # if i ==0:
            #     break
        
        # visulizing images
        # print(np.array(images).shape) #shape of appended array

        # imageio.imwrite("/home/sidra/Documents/image.png", images[0][:,:,127])
        # imageio.imwrite("/home/sidra/Documents/label.png", labels[0][:,:,127])
        # print("image_len", len(images))
   
        images, labels = self.unify_sizes(images, labels)
  
        # print("after unifying ", images[0].shape, labels[0].shape)
  
        # print("vstack shape", np.vstack(images)[2459].shape)   # vsatck stacking vertically. first dim will be (number of image* first dim of
        # asd 
        self.data = np.expand_dims(np.vstack(images), axis=1)
        self.label = np.expand_dims(np.vstack(labels), axis=1)
        self.voxel_dim = np.vstack(self.voxel_dim)
        # print(type(self.data), type(self.label), type(self.voxel_dim))

        # print("image_shape", self.data.shape, "label_shape", self.label.shape, "voxel_shape", self.voxel_dim.shape)
        # print("self.data.shape", self.data.shape)
        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.voxel_dim = torch.from_numpy(self.voxel_dim)
        # print(self.voxel_dim)
        # asd
        
       
        # print("self.data", self.data.shape, "self.label:", self.label.shape,
        #   "self.voxel_dim", self.voxel_dim.shape, "len", len(self.voxel_dim.shape))
        # asd
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]
        # asd
        return data, labels, voxel_dim






class cc359_3d_volume(Dataset):
    def __init__(self, config, train = True, rotate=True, scale=True ):
        self.rotate = rotate
        self.scale = scale
        self.fold = config.fold
        self.train = train
        # self.data = self.load_data()
        self.site = config.site
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


    def load_files(self, data_path):
        self.sagittal = True

        if self.source == "True" and self.train:

            self.images_path = os.path.join(data_path, 'Original', self.folder, "train")
            print("train_path ", self.images_path )
        

        elif self.source == "True" and not self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val")
            print("val_path ",self.images_path)


        elif self.source == "False":
            self.images_path = os.path.join(data_path, 'Original', self.folder, "test.csv")
            print("test_path ", self.images_path)
    
        # Get a list of all the volume data files in the root directory        
        # self.volume_files = np.array(sorted([os.path.join(self.images_path, f) for f in os.listdir(self.images_path) if f.endswith('.nii.gz')]))
        self.volume_files = pd.read_csv(self.images_path).values.ravel().tolist()
        # embed()
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
        sizes = np.zeros(len(input_images), np.int)
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
        
        # Load the volume data from the NIfTI file using nibabel
        img = nib.load(self.volume_files[idx]).get_fdata('unchanged', dtype=np.float32)       
        nib_file = nib.load(self.volume_files[idx])
        
        spacing = [nib_file.header.get_zooms()] * nib_file.shape[0]
        self.voxel_dim= np.array(spacing)
        # embed()
        img = self.img_transform(img)  
        
        # print("file_name", self.volume_files[idx])
        lbl = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, self.volume_files[idx][:-7].split("/")[-1] + '_ss.nii.gz')).get_fdata('unchanged', dtype=np.float32)
        lbl = self.img_transform(lbl)  
        
        
        images, labels = self.unify_sizes(img, lbl)
        data = torch.from_numpy(np.expand_dims(images.copy(), axis=1))
        label = torch.from_numpy(np.expand_dims(labels.copy(), axis=1))
        voxel_dim = torch.from_numpy(self.voxel_dim)
     
        
        return data, label, voxel_dim

