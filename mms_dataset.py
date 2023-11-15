# Loader for the M&Ms dataset
# author: Rasha Sheikh

import numpy as np
import os
from collections import namedtuple
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from IPython import embed


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MMSDataset(Dataset):

    def __init__(self, config, site, train=True):

        self.data_path = config.data_path
        self.fold = config.fold
        self.train = train
        self.refine = config.refine
        self.source = config.source
        self.vendor = site
        self.one_hot_encoding = True
        self.n_classes = 4
    
        
        self.load_dataset_information()
        self.filter_data()
        self.load_files()
              
    def load_dataset_information(self):
        self.meta_info = {}
        file_path = os.path.join(self.data_path, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
        with open(file_path) as f:
            file_content = f.readlines()

        header = file_content[0].strip().split(',')
        header[1] = header[1].replace(' ', '_')
        Meta = namedtuple('Meta', header[1:])
       
        for line in file_content[1:]:
            sample = line.strip().split(',')
            self.meta_info[sample[1]] = Meta(*sample[1:])


                             
    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape
        b = (max_size[0] - current_size[0]) // 2
        a = max_size[0]-(b+current_size[0])
        d = (max_size[1] - current_size[1]) // 2
        c = max_size[1]-(d+current_size[1])
        return np.pad(data_array, ((b,a),(d,c),(0,0)), mode='edge') 
    
                
    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros((len(input_images),2), int)
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
        
        if self.train and self.refine == "False": 
         
            self.images_path = os.path.join(self.data_path, 'Training', 'Labeled')
            all_files = os.listdir(self.images_path)
  

        elif self.refine == "True" and self.train:
            files_list = os.path.join(self.data_path, "_".join(["train", self.vendor]))+ ".csv"
            all_files = pd.read_csv(files_list).values.ravel().tolist()
            
            if self.vendor == "C" or self.vendor == "D":
                self.images_path = os.path.join(self.data_path, 'Training' , "Unlabeled")
            else: 
          
                self.images_path = os.path.join(self.data_path, 'Training', 'Labeled')
                # self.images_path = os.path.join(self.data_path, 'Testing')
        print(f"image path: {self.images_path}, vendor: {self.vendor}")
        

        files = []
        for f in all_files:
            if self.meta_info[f].Vendor == self.vendor:
                files.append(f)
        files = np.array(sorted(files))
        self.files = files
        # embed()

    # def make_weights_for_balanced_classes(self, labels):

    #     count = [0] * self.n_classes
    #     for lbl in range(len(labels)):                                                         
                                                                                        
    #         unique_values, counts = np.unique(labels[lbl], return_counts=True)                                          
    #         for value, count_value in zip(unique_values, counts):                                   
    #             count[int(value)] += count_value   
        
    #     weight_per_class = [0.] * self.n_classes
    #     N = float(sum(count))           

    #     for i in range(self.n_classes):                                                   
    #         weight_per_class[i] = N/float(count[i])      
    #     print("weights per class",   weight_per_class )    
        
    #     return weight_per_class

 
    
    
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
            # embed()
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
            # embed()
      
            self.voxel_dim.append(np.array(spacing))
 
        images, labels = self.unify_sizes(images, labels)
    
        self.voxel_dim = np.vstack(self.voxel_dim)
     
        self.data = np.expand_dims(np.moveaxis(np.concatenate(images, axis=-1),-1,0), axis=1)
        labels = np.moveaxis(np.concatenate(labels, axis=-1),-1,0)
        # # self.weights = self.make_weights_for_balanced_classes(labels)
        # self.weights = torch.tensor([self.weights])
        # print("check labels")
        # embed()
     
        if self.one_hot_encoding:
            shape = labels.shape
            self.label = np.zeros((shape[0], self.n_classes, shape[1], shape[2]), dtype=np.int_)
            for c in range(self.n_classes):
                self.label[:,c,:,:] = labels[:,:,:]==c

            # self.weights = self.make_weights_for_balanced_classes()
            
        else:
            self.label = labels.astype(np.int_)
            self.label = np.expand_dims(self.label, axis=1)


        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        
        self.voxel_dim = torch.from_numpy(self.voxel_dim)
        # embed()


         


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]
        
        return data, labels, voxel_dim #, self.weights
    


class mms_3d_volume(Dataset):

    def __init__(self, config, site, train=True):
  
        self.data_path = config.data_path
        self.fold = config.fold
        self.train = train
        self.refine = config.refine
        self.source = config.source
        self.vendor = site
        self.one_hot_encoding = True
        self.n_classes = 4
        
        self.load_dataset_information()
        self.filter_data()
        self.load_files()
        
        
    def load_dataset_information(self):
        self.meta_info = {}
        file_path = os.path.join(self.data_path, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
        with open(file_path) as f:
            file_content = f.readlines()

        header = file_content[0].strip().split(',')
        header[1] = header[1].replace(' ', '_')
        Meta = namedtuple('Meta', header[1:])
        for line in file_content[1:]:
            sample = line.strip().split(',')
            self.meta_info[sample[1]] = Meta(*sample[1:])
         
    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape
        b = (max_size[0] - current_size[0]) // 2
        a = max_size[0]-(b+current_size[0])
        d = (max_size[1] - current_size[1]) // 2
        c = max_size[1]-(d+current_size[1])
        return np.pad(data_array, ((b,a),(d,c),(0,0)), mode='edge') 
              
    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros((len(input_images),2), dtype=np.int32)
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
          
        if self.train and  self.refine == "False":
            print("in base model")
            self.images_path = os.path.join(self.data_path, 'Training', 'Labeled')
            all_files = np.array(sorted(os.listdir(self.images_path)))
            
        elif self.refine == "True" and self.train:

            files_list = os.path.join(self.data_path, "_".join(["train", self.vendor]))+ ".csv"
            all_files = pd.read_csv(files_list).values.ravel().tolist()
            
            if self.vendor == "C":
                self.images_path = os.path.join(self.data_path, 'Training' , "Unlabeled")
            else: 
                self.images_path = os.path.join(self.data_path, 'Training', 'Labeled')
            
        elif (self.refine == "True" or self.source == "True") and not self.train:        
            self.images_path = os.path.join(self.data_path, 'Validation')
            all_files = np.array(sorted(os.listdir(self.images_path)))
        
        else:
 
            self.images_path = os.path.join(self.data_path, 'Testing')
            all_files = np.array(sorted(os.listdir(self.images_path)))

        print(f"image path: {self.images_path}, vendor: {self.vendor}")
       
        files = []
        for i, f in enumerate(all_files):
            if self.meta_info[f].Vendor == self.vendor:
                files.append(f)
                
        files = np.array(sorted(files))
        # print(files)
        self.files = files

        # embed()

    def make_weights_for_balanced_classes(self, labels):
        

        # labels = [np.where(label == 0, 1, label) for label in labels] 

        # labels = [np.where(label != 0, 1, label) for label in labels]
        # n_classes = 2 # for binary
        total_counts = [0] * self.n_classes  # Initialize with the number of classes
        unique_counts_per_volume = []

        for lbl in range(len(labels)):                                                              
            
            # Create a list to store unique class counts for this volume                        
            volume_unique_counts = [0] * self.n_classes                                             
                                                                                        
            # Iterate through each 2D slice within the current volume                               
            for slice_2d in labels[lbl]:                                                            
            # Calculate unique values and their counts for the current 2D slice                 
                unique_values, counts = np.unique(slice_2d, return_counts=True)                     
                                                                                                
            # Accumulate counts for each class within the 2D slice                              
                for value, count_value in zip(unique_values, counts):                               
                    volume_unique_counts[int(value)] += count_value                                 
            # Append the unique counts for this volume to the list                              
            unique_counts_per_volume.append(volume_unique_counts)                                   
                                                                                                
            # Accumulate counts for the entire dataset                                              
            for value, count_value in enumerate(volume_unique_counts):                              
                total_counts[value] += count_value  
        
        # weight_per_class = [0.] * 2 # for binary

        # embed()
        weight_per_class = [0.] * self.n_classes 
        N = float(sum(total_counts))  
        # embed()         
        for i in range(self.n_classes):  
        # for i in range(n_classes):   #binary                                                
            weight_per_class[i] = N/float(total_counts[i])      
        print("weights per class",   weight_per_class )

        return weight_per_class 
    
    def load_files(self):

        scaler = MinMaxScaler()
        images = []
        labels = []
        self.voxel_dim = []
  
        for i, f in enumerate(self.files):
            # print("file,name", f)
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

        self.data = [np.rollaxis(image, -1, 0) for image in images]
        labels = [np.rollaxis(lbl, -1, 0) for lbl in labels]
        # print("check labels")
        # embed()
        weights = self.make_weights_for_balanced_classes(labels)
        self.weights = torch.tensor(weights).float()
        
        # print(f'weights, {self.weights}')


        if self.one_hot_encoding:
            self.labels = []
            for i in range(len(labels)):
                label = np.zeros((labels[i].shape[0], self.n_classes, labels[i].shape[1], labels[i].shape[2]), dtype=np.int_)
                for c in range(self.n_classes):
                    label[:, c, :, :] = labels[i][:, :, :] == c
                self.labels.append(label)


        else:
            self.label = labels.astype(np.int_)
            self.label = np.expand_dims(self.label, axis=1)


        self.data = [torch.from_numpy(np.expand_dims(np.asarray(data), axis =1)) for data in self.data]
        self.label = [torch.from_numpy(np.asarray(data)) for data in self.labels]
        self.voxel_dim = [torch.from_numpy(np.asarray(data)) for data in self.voxel_dim]


          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]
       
        return data, labels, voxel_dim , self.weights