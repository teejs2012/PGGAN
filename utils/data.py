# -*- coding: utf-8 -*-
import os, scipy.misc
from glob import glob
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms

#prefix = 'C:\\Users\\yuan\\Downloads'
# prefix = '/Users/yuan/Downloads/'
prefix = './datasets/'

def get_img(img_path, is_crop=True, crop_h=256, resize_h=64, normalize=False):
    img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    resize_w = resize_h
    if is_crop:
        crop_w = crop_h
        h, w = img.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
    else:
        cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
    if normalize:
        cropped_image = cropped_image/127.5 - 1.0
    return np.transpose(cropped_image, [2, 0, 1])


# class CelebA():
#     def __init__(self):
#         datapath = os.path.join(prefix, 'celeba/aligned')
#         self.channel = 3
#         self.data = glob(os.path.join(datapath, '*.jpg'))

#     def __call__(self, batch_size, size):
#         batch_number = len(self.data)/batch_size
#         path_list = [self.data[i] for i in np.random.randint(len(self.data), size=batch_size)]
#         file_list = [p.split('/')[-1] for p in path_list]
#         batch = [get_img(img_path, True, 178, size, True) for img_path in path_list]
#         batch_imgs = np.array(batch).astype(np.float32)
#         return batch_imgs

#     def save_imgs(self, samples, file_name):
#         N_samples, channel, height, width = samples.shape
#         N_row = N_col = int(np.ceil(N_samples**0.5))
#         combined_imgs = np.ones((channel, N_row*height, N_col*width))
#         for i in range(N_row):
#             for j in range(N_col):
#                 if i*N_col+j < samples.shape[0]:
#                     combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
#         combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
#         scipy.misc.imsave(file_name+'.png', combined_imgs)

class Data:
    def __init__(self,folder,max_size=256):
        self.datapath = folder
        if not os.path.isdir(folder):
            print("the folder does not exist")
        self.files = os.listdir(self.datapath)
        np.random.shuffle(self.files)
        self.count = 0
        self.max_size = max_size
    
    def get_count(self):
        return len(self.files)
    
    def next(self, batch_size, res, cur_level):
        if self.count+batch_size >= len(self.files):
            np.random.shuffle(self.files)
            self.count = 0
        imgs = []
        for ind in range(self.count,self.count+batch_size):
            file = self.files[ind]
            file_path = os.path.join(self.datapath,file)
            img = Image.open(file_path)

            transform = transforms.Compose([
                transforms.Resize([res,res]),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(0.1,0.1,0.1,0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
            ])
            result_img = transform(img)
            if cur_level != int(cur_level):
                lower_res_ratio = int(cur_level + 1) - cur_level
                lower_res = res//2
                low_res_img = img.resize((lower_res, lower_res))
                low_res_img = transform(low_res_img)
                result_img = low_res_img * lower_res_ratio + result_img * (1-lower_res_ratio)
                
#             result_img = transform(img)
            result_img = result_img.unsqueeze(0)
            if ind == self.count:
                imgs = result_img
            else:
                imgs = torch.cat((imgs,result_img),0)
                
            # max_result_img = max_transform(img)
            # max_result_img = max_result_img.unsqueeze(0)
            # if ind == self.count:
            #     max_imgs = max_result_img
            # else:
            #     max_imgs = torch.cat((max_imgs,max_result_img),0)
                
        self.count = self.count + batch_size
        return imgs

# class CelebA():
#     def __init__(self):
#         datapath = 'celeba-hq-1024x1024.h5'
#         resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
#                         'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
#         self._base_key = 'data'
#         self.dataset = h5py.File(os.path.join(prefix, datapath), 'r')
#         self._len = {k:len(self.dataset[k]) for k in resolution}
#         assert all([resol in self.dataset.keys() for resol in resolution])
#
#     def __call__(self, batch_size, size, level=None):
#         key = self._base_key + '{}x{}'.format(size, size)
#         idx = np.random.randint(self._len[key], size=batch_size)
#         batch_x = np.array([self.dataset[key][i]/127.5-1.0 for i in idx], dtype=np.float32)
#         if level is not None:
#             if level != int(level):
#                 min_lw, max_lw = int(level+1)-level, level-int(level)
#                 lr_key = self._base_key + '{}x{}'.format(size//2, size//2)
#                 low_resol_batch_x = np.array([self.dataset[lr_key][i]/127.5-1.0 for i in idx], dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
#                 batch_x = batch_x * max_lw + low_resol_batch_x * min_lw
#         return batch_x
#
#     def save_imgs(self, samples, file_name):
#         N_samples, channel, height, width = samples.shape
#         N_row = N_col = int(np.ceil(N_samples**0.5))
#         combined_imgs = np.ones((channel, N_row*height, N_col*width))
#         for i in range(N_row):
#             for j in range(N_col):
#                 if i*N_col+j < samples.shape[0]:
#                     combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
#         combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
#         scipy.misc.imsave(file_name+'.png', combined_imgs)


class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian'):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        self.generator_map = {'gaussian': np.random.randn, 'uniform': np.random.uniform}
        if self.noise_type == 'gaussian':
            self.generator = lambda s: np.random.randn(*s)
        elif self.noise_type == 'uniform':
            self.generator = lambda s: np.random.uniform(-1, 1, size=s)

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)
