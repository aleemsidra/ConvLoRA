# Loader for the Calgary Campinas dataset
# Author: Rasha Sheikh

import numpy as np
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
        self.voxel_dim = [] # ?
     
        images_path = os.path.join(data_path, 'Original', self.folder)
        # print("images_path", images_path )
        
        files = np.array(sorted(os.listdir(images_path)))
        # print("length ", len(files))
     
        if self.fold > 0:
       
            files = self.get_fold(files)
            # print("files", files)
            # print("len", len(files))
          
        
        # if len(self.subj_index) > 0:
        #     # print(type(self.subj_index))
        #     files = files[self.subj_index]
            # print("files", files)
            # print("len", len(files))
            # asd
        # print("self.subj_index", self.subj_index, "len", len(self.subj_index))
        # asd
        # print(files)
        for i, f in enumerate(files):
            # asd
            print("file_name", f)
            nib_file = nib.load(os.path.join(images_path, f))
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
        
            if i ==0:
                break
        
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
        
       
        print("self.data", self.data.shape, "self.label:", self.label.shape,
              "self.voxel_dim", self.voxel_dim.shape, "len", len(self.voxel_dim.shape))
        # asd
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]
        # asd
        return data, labels, voxel_dim



class cc359_volume(Dataset):
    # def __init__(self, data_path, site=2, train=True, fold=-1, rotate=True, scale=True, subj_index=[]):
    def __init__(self, config, train = True, rotate=True, scale=True ): #, (subj_index=[])
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
        self.equalize_slices(self.files)


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
    

    import numpy as np

 
    def equalize_slices(self, files, label = False, voxel = False):
         
        """
        Load data
        """
        train_data = []
        for i, file in enumerate( files):
                if file.endswith('.nii.gz'):
                    if not label: 
                        img = nib.load(os.path.join(self.images_path, file)).get_fdata(dtype=np.float32)
                        if voxel:
                           img = [nib.load(os.path.join(self.images_path, file)).header.get_zooms()] * img.shape[0]

                    else:
                        img = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, file[:-7] + '_ss.nii.gz')).get_fdata(
                        'unchanged', dtype=np.float32)

                    train_data.append(img)

      
        """
        Equalizes the number of slices in each 3D volume of a list of tensors.
        """
        # embed()
        # Find the maximum number of slices in a 3D volume
        max_slices = np.max([tensor.shape[0] for tensor in train_data])
        
        # Create a new list with the same number of tensors as the original list,
        # but with the number of slices equalized in each 3D volume
        new_list = []
        for tensor in train_data:
            num_slices = tensor.shape[0]
            new_tensor = np.zeros((max_slices, tensor.shape[1], tensor.shape[2]))
            new_tensor[:num_slices,:,:] = tensor
            new_list.append(new_tensor)
      
        return new_list



    def load_files(self, data_path):
       
        self.sagittal = True

     
        if self.source and self.train:

            self.images_path = os.path.join(data_path, 'Original', self.folder, "train")
            print("image_path ", self.images_path )
        

        elif self.source and not self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val")
            print("image_path ",self.images_path)


        else:
            print(self.source, self.train)
            self.images_path = os.path.join(data_path, 'Original', self.folder)
            print("image_path ", self.images_path)

       
        self.files = np.array(sorted(os.listdir(self.images_path)))
        self.data = cc359_volume.equalize_slices(self, self.files)
        self.label = cc359_volume.equalize_slices(self, self.files, label = True)
        # self.voxel = cc359_volume.equalize_slices(self, self.files, voxel = True)
       
  
    def __len__(self):
        return len(self.files)

    # def __getitem__(self, idx):

    #     print("file_name", os.path.join(self.images_path, self.files[idx]))

    #     nib_file = nib.load(os.path.join(self.images_path, self.files[idx]))
    #     img = nib_file.get_fdata('unchanged', dtype=np.float32) 

    #     if not self.sagittal:
    #         img = np.moveaxis(img, -1, 0)
        
    #     if self.rotate:
    #         img = np.rot90(img, axes=(1, 2))

    #     if img.shape[1] != img.shape[2]:
    #         self.img = self.pad_image(img)

    #     data = torch.from_numpy(np.expand_dims(img.copy(), axis=1))
    
    #     lbl = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, self.files[idx][:-7] + '_ss.nii.gz')).get_fdata(
    #             'unchanged', dtype=np.float32)

    #     if not self.sagittal:
    #             lbl = np.moveaxis(lbl, -1, 0)
    #     if self.rotate:
    #         lbl = np.rot90(lbl, axes=(1, 2))
    #     if lbl.shape[1] != lbl.shape[2]:
    #         lbl = self.pad_image(lbl)
        
    #     label = torch.from_numpy(np.expand_dims(lbl.copy(),axis=1))
    #     # print("label", label.shape)
    #     spacing = [nib_file.header.get_zooms()] * img.shape[0]
    #     voxel_dim = np.array(spacing) 
     
    #     return data , label , voxel_dim
    

    def __getitem__(self, idx):
        print("file_name", os.path.join(self.images_path, self.files[idx])) # for get voxel from header

        nib_file = nib.load(os.path.join(self.images_path, self.files[idx]))
        # img = nib_file.get_fdata('unchanged', dtype=np.float32) 
        img = self.data[idx]

        if not self.sagittal:
            img = np.moveaxis(img, -1, 0)
        
        if self.rotate:
            img = np.rot90(img, axes=(1, 2))

        if img.shape[1] != img.shape[2]:
            self.img = self.pad_image(img)
        
        # for i in range(img.shape[0]):

        #     data_list.append(np.expand_dims(img[i], axis=0))
        
        # data = np.concatenate(data_list, axis=0)
        data = torch.from_numpy(np.expand_dims(img.copy(), axis=1))
          
        label = self.label[idx]

        if not self.sagittal:
            img = np.moveaxis(label, -1, 0)
        
        if self.rotate:
            label = np.rot90(label, axes=(1, 2))

        if label.shape[1] != label.shape[2]:
            label = self.pad_image(img)

        label = torch.from_numpy(np.expand_dims(label.copy(),axis=1))
        spacing = [nib_file.header.get_zooms()] * img.shape[0]
        voxel_dim = np.array(spacing) 


        return data , label,  voxel_dim


    