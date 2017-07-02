""" Kaggle Sealion Pytorch Dataset
Pytorch Dataset code for patched based training and prediction of the
NOAA Fishes Sea Lion counting Kaggle data.

Dataset code generates or loads targets for density and counception
based counting models.
"""
from collections import defaultdict
import cv2
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
from PIL import Image
import random
import pandas as pd
import numpy as np
import os
import functools
import time
import mytransforms
import utils

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
LABEL_TYPE = ['all', 'ground-cover', 'sky-cover', 'primary']

LABEL_ALL = [
    'blow_down',
    'conventional_mine',
    'slash_burn',
    'blooming',
    'artisinal_mine',
    'selective_logging',
    'bare_ground',
    'cloudy',
    'haze',
    'habitation',
    'cultivation',
    'partly_cloudy',
    'water',
    'road',
    'agriculture',
    'clear',
    'primary',
]

LABEL_GROUND_COVER = [
    'blow_down',
    'conventional_mine',
    'slash_burn',
    'blooming',
    'artisinal_mine',
    'selective_logging',
    'bare_ground',
    'habitation',
    'cultivation',
    'water',
    'road',
    'agriculture',
    'primary',
]

LABEL_SKY_COVER = [
    'cloudy',
    'haze',
    'partly_cloudy',
    'clear',
]

LABEL_PRIMARY = ['primary']


def to_tensor(arr):
    assert(isinstance(arr, np.ndarray))
    t = torch.from_numpy(arr.transpose((2, 0, 1)))
    if isinstance(t, torch.ByteTensor):
        return t.float().div(255)
    return t


def find_inputs(folder, types=IMG_EXTENSIONS, extract_extra=False):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs


class AmazonDataset(data.Dataset):
    def __init__(
            self,
            input_root,
            label_file='',
            label_type='all',
            multi_label=True,
            train=True,
            fold=0,
            img_type='.jpg',
            img_size=(256, 256),
            per_image_norm=False,
            transform=None):

        assert img_type in ['.jpg', '.tif']
        inputs = find_inputs(
            input_root, types=[img_type], extract_extra=False)
        if len(inputs) == 0:
            raise (RuntimeError("Found 0 images in : " + input_root))

        if label_file:
            label_df = pd.read_csv(label_file)
            if train:
                label_df = label_df[label_df['fold'] != fold]
            else:
                label_df = label_df[label_df['fold'] == fold]
            label_df.drop(['fold'], 1, inplace=True)

            input_dict = dict(inputs)
            print(len(input_dict), len(label_df.index))
            label_df = label_df[label_df.image_name.map(lambda x: x in input_dict)]
            label_df['filename'] = label_df.image_name.map(lambda x: input_dict[x])
            self.inputs = label_df['filename'].tolist()
            print(len(self.inputs), len(label_df.index))

            if label_type == 'all':
                self.label_array = label_df.as_matrix(columns=LABEL_ALL).astype(np.float32)
            elif label_type == 'ground_cover':
                self.label_array = label_df.as_matrix(columns=LABEL_GROUND_COVER).astype(np.float32)
            elif label_type == 'sky_cover':
                self.label_array = label_df.as_matrix(columns=LABEL_SKY_COVER).astype(np.float32)
            elif label_type == 'primary':
                self.label_array = label_df.as_matrix(columns=LABEL_PRIMARY).astype(np.float32)
            else:
                assert False and 'Invalid label type'

            if not multi_label:
                self.label_array = np.argmax(self.label_array, axis=1)

            self.label_array = torch.from_numpy(self.label_array)
        else:
            assert not train
            self.label_array = None
            self.inputs = [x[1] for x in inputs]

        self.train = train
        if img_type == '.jpg':
            self.dataset_mean = [0.31535792, 0.34446435, 0.30275137]
            self.dataset_std = [0.05338271, 0.04247036, 0.03543708]
        else:
            #self.dataset_mean = [4988.75696302, 4270.74552695, 3074.87909779, 6398.84897763]
            #self.dataset_std = [399.06597519, 408.51461036, 453.1910904, 858.46477922]
            self.dataset_mean = [6398.84897763, 4988.75696302, 4270.74552695] # NRG
            self.dataset_std = [858.46477922, 399.06597519, 408.51461036] # NRG

        self.img_size = img_size
        self.img_type = img_type
        if transform is None:
            tfs = []
            if img_type == '.jpg':
                tfs.append(mytransforms.ToTensor())
                if self.train:
                    tfs.append(mytransforms.ColorJitter())
                if not per_image_norm:
                    tfs.append(transforms.Normalize(self.dataset_mean, self.dataset_std))
            else:
                tfs.append(mytransforms.NormalizeImgIn64(self.dataset_mean, self.dataset_std))
                tfs.append(mytransforms.ToTensor())
                #if self.train:
                #    tfs.append(mytransforms.ColorJitter())
            self.transform = transforms.Compose(tfs)

    def _load_input(self, index):
        path = self.inputs[index]
        #print("Loading %s" % path)
        if self.img_type == '.jpg':
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            img = cv2.imread(path, -1) # loaded as BGRN
            img_nrg = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img_nrg[:, :, 0] = img[:, :, 3]  # N
            img_nrg[:, :, 1] = img[:, :, 0]  # R
            img_nrg[:, :, 2] = img[:, :, 1]  # G
            return img_nrg

    def _random_crop_and_transform(self, input_img, rot=0.0):
        angle = 0.
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        do_rotate = (rot > 0 and random.random() < 0.25) if not hflip and not vflip else False
        h, w = input_img.shape[:2]
        attempts = 0
        while attempts < 3:
            if do_rotate:
                angle = random.uniform(-rot, rot)
            scale = random.uniform(0.88, 1.14)
            crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], angle, scale)
            if crop_w <= w and crop_h <= h:
                break
        if crop_w > w or crop_h > h:
            print('Crop %d, %d too large with rotation %f, skipping rotation.' % (crop_w, crop_h, angle))
            angle = 0.0
            crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], angle, scale)

        #print('hflip: %d, vflip: %d, angle: %f, scale: %f' % (hflip, vflip, angle, scale))
        hd = max(0, h - crop_h)
        wd = max(0, w - crop_w)
        ho = random.randint(0, hd) - hd // 2
        wo = random.randint(0, wd) - wd // 2
        cx = w // 2 + wo
        cy = h // 2 + ho
        #print(crop_w, crop_h, cx, cy, wd, hd, wo, ho)
        input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)

        # Perform tile geometry transforms if needed
        if angle or scale != 1. or hflip or vflip:
            Mtrans = np.identity(3)
            Mtrans[0, 2] = (self.img_size[0] - crop_w) // 2
            Mtrans[1, 2] = (self.img_size[1] - crop_h) // 2
            if hflip:
                Mtrans[0, 0] *= -1
                Mtrans[0, 2] = self.img_size[0] - Mtrans[0, 2]
            if vflip:
                Mtrans[1, 1] *= -1
                Mtrans[1, 2] = self.img_size[1] - Mtrans[1, 2]

            if angle or scale != 1.:
                Mrot = cv2.getRotationMatrix2D((crop_w//2, crop_h//2), angle, scale)
                Mfinal = np.dot(Mtrans, np.vstack([Mrot, [0, 0, 1]]))
            else:
                Mfinal = Mtrans

            input_img = cv2.warpAffine(input_img, Mfinal[:2, :], self.img_size)

        return input_img

    def _centre_crop_and_scale(self, input_img, scale=1.0):
        h, w = input_img.shape[:2]
        cx = w // 2
        cy = h // 2
        crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], scale=scale)
        #print(crop_w, crop_h)
        input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)
        if scale != 1.0:
            input_img = cv2.resize(input_img, self.img_size)
        #print(input_img.shape)
        return input_img

    def __getitem__(self, index):
        #input_path = self.inputs[index]
        input_img = self._load_input(index)
        if self.label_array is not None:
            label_tensor = self.label_array[index]
        #h, w = input_img.shape[:2]
        if self.train:
            input_img = self._random_crop_and_transform(input_img, rot=5.0)
            input_tensor = self.transform(input_img)
        else:
            input_img = self._centre_crop_and_scale(input_img, scale=0.934)
            input_tensor = self.transform(input_img)

        index_tensor = torch.LongTensor([index])
        #print(input_tensor.size(), label_tensor)

        return input_tensor, label_tensor, index_tensor

    def __len__(self):
        return len(self.inputs)

