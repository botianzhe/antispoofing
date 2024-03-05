import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import segmentation_models as smp
from PIL import Image
from torchvision import transforms
import torch.nn as nn
def load_model(model, path):
    ckpt = torch.load(path, map_location="cpu")
    # print(ckpt)
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("acc1", 0.0)
    model.load_state_dict(ckpt["state_dict"])
    return model

def buildmodel():
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation=None,      # activation function, default is None
        classes=2,                 # define number of output labels
    )
    unet = smp.Unet(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet

modelpath1='p1_eff0_20.pth'
modelpath21='p21_eff0_14.pth'
modelpath22='p22_eff0_12.pth'
gunet=buildmodel()
gunet=load_model(gunet,modelpath1)
modelname=gunet.name
gunet = gunet.cuda()

gunet21=buildmodel()
gunet21=load_model(gunet21,modelpath21)
modelname=gunet21.name
gunet21 = gunet21.cuda()

gunet22=buildmodel()
gunet22=load_model(gunet22,modelpath22)
modelname=gunet22.name
gunet22 = gunet22.cuda()

torch.autograd.set_grad_enabled(False)
gunet.eval()
gunet21.eval()
gunet22.eval()
datapath='/mnt/UserData1/Data/FaceAntiSpoofing/'
with open(datapath+'p1/dev.txt','r') as f:
    data=f.read()
evaldata1=data.split('\n')[:-1]

with open(datapath+'p2.1/dev.txt','r') as f:
    data=f.read()
evaldata2=data.split('\n')[:-1]

with open(datapath+'p2.2/dev.txt','r') as f:
    data=f.read()
evaldata3=data.split('\n')[:-1]

datapath2='/mnt/UserData1/Data/FaceAntiSpoofingTest/phase2/'
with open(datapath2+'p1/test.txt','r') as f:
    data=f.read()
evaldata12=data.split('\n')[:-1]

with open(datapath2+'p2.1/test.txt','r') as f:
    data=f.read()
evaldata22=data.split('\n')[:-1]

with open(datapath2+'p2.2/test.txt','r') as f:
    data=f.read()
evaldata32=data.split('\n')[:-1]
print(len(evaldata1),len(evaldata12),len(evaldata2),len(evaldata22),len(evaldata3),len(evaldata32))

torch.set_printoptions(precision=6,sci_mode=False)
np.set_printoptions(suppress=True)
for data in [evaldata1,evaldata12]:
    for file in data:
        if 'test' in file:
            filepath=datapath2+file
            filepath=filepath.replace("FaceAntiSpoofingTest",'FaceAntiSpoofingTest_Face')
        else:
            filepath=datapath+file
            filepath=filepath.replace('FaceAntiSpoofing','FaceAntiSpoofing_Face')
        img= Image.open(filepath).convert('RGB')
        img=np.array(img.resize((256,256)))
        img = img[:, :, ::-1].copy()
        img=transforms.ToTensor()(img)
        img = img.cuda(non_blocking=True)
        _,clspred = gunet(img.unsqueeze(0))
        clspred=nn.Softmax(dim=1)(clspred)
        score=clspred.cpu().detach().numpy()[0][1]
        print(file.replace('p1/','').replace('p2.1/','').replace('p2.2/',''),f"{score:.6f}")

for data in [evaldata2,evaldata22]:
    for file in data:
        if 'test' in file:
            filepath=datapath2+file
            filepath=filepath.replace("FaceAntiSpoofingTest",'FaceAntiSpoofingTest_Face')
        else:
            filepath=datapath+file
            filepath=filepath.replace('FaceAntiSpoofing','FaceAntiSpoofing_Face')
        img= Image.open(filepath).convert('RGB')
        img=np.array(img.resize((256,256)))
        img = img[:, :, ::-1].copy()
        img=transforms.ToTensor()(img)
        img = img.cuda(non_blocking=True)
        _,clspred = gunet21(img.unsqueeze(0))
        clspred=nn.Softmax(dim=1)(clspred)
        score=clspred.cpu().detach().numpy()[0][1]
        print(file.replace('p1/','').replace('p2.1/','').replace('p2.2/',''),f"{score:.6f}")

for data in [evaldata3,evaldata32]:
    for file in data:
        if 'test' in file:
            filepath=datapath2+file
            filepath=filepath.replace("FaceAntiSpoofingTest",'FaceAntiSpoofingTest_Face')
        else:
            filepath=datapath+file
            filepath=filepath.replace('FaceAntiSpoofing','FaceAntiSpoofing_Face')
        img= Image.open(filepath).convert('RGB')
        img=np.array(img.resize((256,256)))
        img = img[:, :, ::-1].copy()
        img=transforms.ToTensor()(img)
        img = img.cuda(non_blocking=True)
        _,clspred = gunet22(img.unsqueeze(0))
        clspred=nn.Softmax(dim=1)(clspred)
        score=clspred.cpu().detach().numpy()[0][1]
        print(file.replace('p1/','').replace('p2.1/','').replace('p2.2/',''),f"{score:.6f}")