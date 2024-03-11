from torchvision import transforms
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,JpegCompression,VerticalFlip,Rotate,
    Transpose, ShiftScaleRotate, Blur,GaussianBlur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,IAAAdditiveGaussianNoise,Resize,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomBrightness,ToSepia,RandomCrop
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            MotionBlur(),
            MedianBlur(),
            GaussNoise(),
        ], p=0.4),
        OneOf([
            MotionBlur(p=0.25),
            JpegCompression(60,100),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
            Rotate()
        ], p=0.4)
    ], p=p)
augmentation=strong_aug()
trans=transforms.Compose([
            
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.38348666],std=[0.20834281])
            # transforms.Normalize(mean=[0.38348666, 0.39193852, 0.4665315],std=[0.20834281, 0.20540032, 0.24183848])
        ])
