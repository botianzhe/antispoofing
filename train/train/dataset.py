import os
import cv2
import random
import json
from PIL import Image
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import dlib
from faker.generate import mask_face

class DFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        
        self.next_epoch()

    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)

        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        # self.dataset=random.sample(self.dataset,5000)
        random.shuffle(self.dataset)
    def __getitem__(self, item):
        sample,label = self.dataset[item]
        label=0 if label==0 else 1
        anchorimg=Image.open(sample).convert('RGB')
        anchorimg=np.array(anchorimg.resize((256,256)))
        anchorimg=anchorimg[:,:,::-1].copy()
        if self.dataselect == 'train':
            anchorimg = self.aug(image=anchorimg)['image']
        anchorimg = self.trans(anchorimg)
        onehot_label=np.zeros(2)
        onehot_label[label]=1
        onehot_label=torch.tensor(onehot_label).float()
        return anchorimg, onehot_label

    def __len__(self):
        return len(self.dataset)
    


class AUGDFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.next_epoch()

    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        with open('jsons/data_devtest.json', 'r') as f:
            evaldata = json.load(f)
        if self.dataselect == 'train':
            if 'p1' in self.jsonpath:
                # trainset = data['train']
                trainset = data['train']+evaldata['p1test']+evaldata['p1dev']
            if 'p2.1' in self.jsonpath:
                # trainset = data['train']
                trainset = data['train']+evaldata['p21test']+evaldata['p21dev']
            if 'p2.2' in self.jsonpath:
                # trainset = data['train']
                trainset = data['train']+evaldata['p22test']+evaldata['p22dev']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        # self.dataset=random.sample(self.dataset,5000)
        random.shuffle(self.dataset)
    def __getitem__(self, item):
        sample,label = self.dataset[item]
        label=0 if label==0 else 1
        anchorimg=Image.open(sample).convert('RGB')
        anchorimg=np.array(anchorimg.resize((256,256)))
        anchorimg=anchorimg[:,:,::-1].copy()
        if self.dataselect == 'train':
            anchorimg = self.aug(image=anchorimg)['image']
            if random.randint(0,1)==0:
                rects = self.detector(np.array(anchorimg), 0)
                if len(rects)>=1:
                    landmarks = np.matrix([[p.x, p.y] for p in self.predictor(np.array(anchorimg),rects[0]).parts()])
                    # print(landmarks.shape)
                    anchorimg,mask = mask_face(np.array(anchorimg),landmarks)
                    # cv2.imwrite(str(item)+'.jpg',anchorimg)
        anchorimg = self.trans(anchorimg)
        onehot_label=np.zeros(2)
        onehot_label[label]=1
        onehot_label=torch.tensor(onehot_label).float()
        return anchorimg, onehot_label

    def __len__(self):
        return len(self.dataset)