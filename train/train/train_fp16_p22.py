import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import DFDataset,AUGDFDataset
from torch import nn, optim
import datetime
from tqdm import tqdm
import imageio
from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score
import segmentation_models as smp
from sklearn.metrics import confusion_matrix
from thop import profile,clever_format
jsonpath = 'jsons/data_p2.2.json'
ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
            print(self.next_data)
        except StopIteration:
            self.next_data = None
            return
        
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is not None:
            data, label = self.next_data
            self.preload()
            return data, label
        else:
            return None, None

def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    filedir=os.path.dirname(path)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if torch.is_tensor(v):
            v = v.cpu()
        new_state_dict[k] = v

    torch.save({
        "epoch": epoch,
        "arch": arch,
        "acc1": acc1,
        "state_dict": new_state_dict,
    }, path)


def load_model(model, path):
    ckpt = torch.load(path, map_location="cpu")
    print(ckpt.keys())
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

modelpath='../p22_eff0_12.pth'
gunet=buildmodel()
modelname='deeplabv3plus'
# gunet=load_model(gunet,modelpath)
# input = torch.randn(1, 3, 256, 256)
# macs, params = profile(gunet, inputs=(input, ))
# macs, params = clever_format([macs, params], "%.3f")
# print(macs,params)

gunet = nn.DataParallel(gunet.cuda())
criterion2 = nn.BCEWithLogitsLoss()
criterion3=nn.BCELoss()
gunet_optimizer = optim.AdamW(gunet.parameters(), 1e-5, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(gunet_optimizer, step_size=1, gamma=0.2)
class_sample_counts = [9000, 5400, 18000,2700]
# train_dataset = DFDataset(
#     '', 'train', trans=trans, augment=augmentation, jsonpath=jsonpath)
train_dataset = AUGDFDataset(
    '', 'train', trans=trans, augment=augmentation, jsonpath=jsonpath)
validate_dataset = DFDataset(
    '', 'val', trans=trans, augment=augmentation, jsonpath=jsonpath)


train_loader = DataLoaderX(train_dataset, batch_size=64,
                           num_workers=8, pin_memory=True,prefetch_factor=3,drop_last=True)
validate_loader = DataLoaderX(
    validate_dataset, batch_size=64, num_workers=8, pin_memory=True,prefetch_factor=3,drop_last=True)
scaler = torch.cuda.amp.GradScaler()
log_dir='saved_models/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_loss = 1
best_auc=0.99
best_epoch = 0
best_model = None
start_epoch = 0

for epoch in range(100):
    gunet.train()
    train_loss = []
    train_acc = []
    train_auc = []
    APCERs=[]
    BPCERs=[]
    ACERs=[]
    with tqdm(train_loader, desc='Batch') as bar:
        count=0
        for b, batch in enumerate(bar):
            anchorimg,label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            gunet_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                _,clspred = gunet(anchorimg)
                # print(clspred.shape,label.shape)
                loss = criterion2(clspred,label)
                clspred=nn.Softmax(dim=1)(clspred)
            scaler.scale(loss).backward()
            scaler.step(gunet_optimizer)
            scaler.update()
            try:
                auc=roc_auc_score(label.cpu(),clspred.detach().cpu())
                cm=confusion_matrix(label.cpu()[:,1].round(),clspred.detach().cpu()[:,1].round())
                TN = cm[0][0]
                FN = cm[1][0]
                TP = cm[1][1]
                FP = cm[0][1]
                APCER = FN / (TP + FN)
                BPCER= FP/(FP + TN)
                ACER = (APCER + BPCER) / 2
            except:
                auc=0.5
                APCER = 100
                BPCER= 100
                ACER = 100
            out = torch.argmax(clspred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()
            
            bar.set_postfix(
                loss=loss.item(),
                batch_acc=batch_acc,
                auc=auc,
                APCER=APCER,
                BPCER=BPCER,
                ACER=ACER
            )
            train_loss.append(loss.item())
            train_acc.append(batch_acc)
            train_auc.append(auc)
            APCERs.append(APCER)
            BPCERs.append(BPCER)
            ACERs.append(ACER)
    epoch_loss = np.mean(train_loss)
    epoch_acc = np.mean(train_acc)
    epoch_auc = np.mean(train_auc)
    epoch_APCERs = np.mean(APCERs)
    epoch_BPCERs = np.mean(BPCERs)
    epoch_ACERs = np.mean(ACERs)
    # scheduler.step()
    
    print(epoch, "Epoch Loss:", epoch_loss,"Epoch ACC:", epoch_acc,"Epoch AUC:", epoch_auc,"Epoch APCERs:", epoch_APCERs,"Epoch BPCERs:", epoch_BPCERs,"Epoch BPCERs:", epoch_ACERs)
    torch.autograd.set_grad_enabled(False)
    gunet.eval()
    val_loss = []
    val_acc = []
    val_auc=[]
    APCERs=[]
    BPCERs=[]
    ACERs=[]
    
    with tqdm(validate_loader, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg,label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            _,clspred = gunet(anchorimg)
            clspred=nn.Softmax(dim=1)(clspred)
            loss = criterion3(clspred,label)
            try:
                auc=roc_auc_score(label.cpu(),clspred.detach().cpu())
                cm=confusion_matrix(label.cpu()[:,1].round(),clspred.detach().cpu()[:,1].round())
                TN = cm[0][0]
                FN = cm[1][0]
                TP = cm[1][1]
                FP = cm[0][1]
                APCER = FN / (TP + FN)
                BPCER= FP/(FP + TN)
                ACER = (APCER + BPCER) / 2
            except:
                auc=0.5
                APCER = 100
                BPCER= 100
                ACER = 100
            out = torch.argmax(clspred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()
            
            bar.set_postfix(
                loss=loss.item(),
                batch_acc=batch_acc,
                auc=auc,
                APCER=APCER,
                BPCER=BPCER,
                ACER=ACER
            )
            val_loss.append(loss.item())
            val_acc.append(batch_acc)
            val_auc.append(auc)
            APCERs.append(APCER)
            BPCERs.append(BPCER)
            ACERs.append(ACER)
    epoch_loss = np.mean(val_loss)
    epoch_acc = np.mean(val_acc)
    epoch_auc = np.mean(val_auc)
    epoch_APCERs = np.mean(APCERs)
    epoch_BPCERs = np.mean(BPCERs)
    epoch_ACERs = np.mean(ACERs)
    torch.autograd.set_grad_enabled(True)
    print(epoch, "Val Epoch Loss:", epoch_loss,"Val Epoch ACC:", epoch_acc,"Val Epoch AUC:", epoch_auc,"Val Epoch APCERs:", epoch_APCERs,"Val Epoch BPCERs:", epoch_BPCERs,"Val Epoch BPCERs:", epoch_ACERs)

    if epoch_loss <= best_loss:
        best_loss = epoch_loss
        best_auc = epoch_auc
        ckpt_path = os.path.join(log_dir,'p22_eff0.pth')
        save_checkpoint(
            ckpt_path,
            gunet.state_dict(),
            epoch=epoch + 1,
            acc1=0)
