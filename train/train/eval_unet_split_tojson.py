import torch
import numpy as np
import os
from strong_transform import augmentation, trans
from sklearn.metrics import roc_auc_score
import segmentation_models as smp
from PIL import Image
from torchvision import transforms
import json
from torch import nn
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
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet

modelpath1='../code/p1_eff0_20.pth'
modelpath21='../code/p21_eff0_14.pth'
modelpath22='../code/p22_eff0_12.pth'

gunet1=buildmodel()
gunet1=load_model(gunet1,modelpath1)
modelname=gunet1.name
gunet1 = gunet1.cuda()

gunet21=buildmodel()
gunet21=load_model(gunet21,modelpath21)
modelname=gunet21.name
gunet21 = gunet21.cuda()

gunet22=buildmodel()
gunet22=load_model(gunet22,modelpath22)
modelname=gunet22.name
gunet22 = gunet22.cuda()

gunet1.eval()
gunet21.eval()
gunet22.eval()
print('Models loaded')

datapath='/mnt/UserData1/peipeng/Data/FaceAntiSpoofing_Face/'
with open(datapath+'p1/dev.txt','r') as f:
    data=f.read()
evaldata1=data.split('\n')[:-1]

with open(datapath+'p2.1/dev.txt','r') as f:
    data=f.read()
evaldata2=data.split('\n')[:-1]

with open(datapath+'p2.2/dev.txt','r') as f:
    data=f.read()
evaldata3=data.split('\n')[:-1]

with open(datapath+'p1/test.txt','r') as f:
    data=f.read()
evaldata12=data.split('\n')[:-1]

with open(datapath+'p2.1/test.txt','r') as f:
    data=f.read()
evaldata22=data.split('\n')[:-1]

with open(datapath+'p2.2/test.txt','r') as f:
    data=f.read()
evaldata32=data.split('\n')[:-1]
print(len(evaldata1),len(evaldata12),len(evaldata2),len(evaldata22),len(evaldata3),len(evaldata32))


data={}
torch.set_printoptions(precision=6,sci_mode=False)
np.set_printoptions(suppress=True)
a1,a12,a2,a22,a3,a32=[],[],[],[],[],[]
index=0
for d in [evaldata1,evaldata12,evaldata2,evaldata22,evaldata3,evaldata32]:
    if index<=1:
        gunet=gunet1
    elif index<=3:
        gunet=gunet21
    else:
        gunet=gunet22
    for file in d:
        filepath=datapath+file
        img= Image.open(filepath).convert('RGB')
        img=np.array(img.resize((256,256)))
        img = img[:, :, ::-1].copy()
        img=transforms.ToTensor()(img)
        img = img.cuda(non_blocking=True)
        _,clspred = gunet(img.unsqueeze(0))
        clspred=nn.Softmax(dim=1)(clspred)
        score=clspred.cpu().detach().numpy()[0][1]
        
        if score >= 0.7:
            score=1
        elif score<=0.3:
            score=0
        else:
            continue
        if 'p1' in file and 'dev' in file:
            a1.append([filepath,score])
        if 'p1' in file and 'test' in file:
            a12.append([filepath,score])
        if 'p2.1' in file and 'dev' in file:
            a2.append([filepath,score])
        if 'p2.1' in file and 'test' in file:
            a22.append([filepath,score])
        if 'p2.2' in file and 'dev' in file:
            a3.append([filepath,score])
        if 'p2.2' in file and 'test' in file:
            a32.append([filepath,score])
        print(file.replace('p1/','').replace('p2.1/','').replace('p2.2/',''),f"{score:.6f}")
    index+=1
data['p1dev']=a1
data['p1test']=a12
data['p21dev']=a2
data['p21test']=a22
data['p22dev']=a3
data['p22test']=a32
with open('jsons/data_devtest.json', 'w') as json_file:
    json.dump(data,json_file)