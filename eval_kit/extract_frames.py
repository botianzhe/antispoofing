import cv2
from face_utils import norm_crop, FaceDetector
import os,sys
import numpy as np
from PIL import Image
from torch import nn, optim
import torch
from torchvision import transforms as T

face_detector = FaceDetector()

face_detector.load_checkpoint("models/RetinaFace-Resnet50-fixed.pth")

def extractface(frame):
    boxes, landms = face_detector.detect(frame)
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # print(areas.argmax())
    order = areas.argmax()
    boxes = boxes[order]
    landms = landms[order]
    # Crop faces
    landmarks = landms.detach().numpy().reshape(5, 2).astype(np.int32)
    img = norm_crop(frame, landmarks, image_size=256)
    return np.array(img)

torch.set_grad_enabled(False)
path='/mnt/UserData1/peipeng/Data/FaceAntiSpoofing/'
num=0
for root,dir,names in os.walk(path):
    for name in names:
        filepath = os.path.join(root, name)  # mp4文件原始地址
        name=filepath.replace(path,'')
        if 'jpg' in filepath:
            num+=1
index=0
for root,dir,names in os.walk(path):
    for name in names:
        filepath = os.path.join(root, name)  # mp4文件原始地址
        if 'jpg' in filepath:
            targetfacepath=filepath.replace("FaceAntiSpoofing",'FaceAntiSpoofing_Face')
            targetfacedir=targetfacepath.replace(name,"")
            print(targetfacedir)
            if not os.path.exists(targetfacedir):
                os.makedirs(targetfacedir)
            img=Image.open(filepath).convert('RGB')
            img=np.array(img)
            try:
                face=extractface(img)
            except:
                face=img
                print(filepath)
            cv2.imwrite(targetfacepath,face)
            print(index,'/',num)
            index+=1
