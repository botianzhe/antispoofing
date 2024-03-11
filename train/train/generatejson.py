import os
import random,json
path='/mnt/UserData1/peipeng/Data/FaceAntiSpoofing_Face/'

def getlabel(labelpath='p1/train_label.txt'):
    with open(path+labelpath,'r') as f:
        data=f.read()
    datalist=data.split('\n')[:-1]
    labeldict={}
    for i in range(len(datalist)):
        name,label=datalist[i].split(' ')
        labeldict[name]=int(label)
    return labeldict

if not os.path.exists('jsons'):
    os.makedirs('jsons')
p1labeldict=getlabel()
p21labeldict=getlabel('p2.1/train_label.txt')
p22labeldict=getlabel('p2.2/train_label.txt')
print(len(p1labeldict),len(p21labeldict),len(p22labeldict))
plabeldict=[p1labeldict,p21labeldict,p22labeldict]
pnames=['p1','p2.1','p2.2']
for index in range(3):
    print(pnames[index])
    traindata=[]
    for root,dir,names in os.walk(path):
        for name in names:
            filepath = os.path.join(root, name)  # mp4文件原始地址
            name=filepath.replace(path,'')
            # print(filepath,name)
            if 'jpg' in filepath and 'train' in filepath:
                if pnames[index] in filepath:     
                    label=plabeldict[index][name]
                    if label==1:
                        label=1
                    else:
                        label=0
                    traindata.append([filepath,label])

    real=[f for f in traindata if f[1]==0]
    fake1=[f for f in traindata if f[1]==1]
    traindata=real+fake1
    trainreal=random.sample(real,int(0.85*len(real)))
    trainfake1=random.sample(fake1,int(0.85*len(fake1)))

    realpath=[d[0] for d in real]
    valreal=[f for f in real if f not in trainreal]
    valfake1=[f for f in fake1 if f not in trainfake1]

    print('train',len(trainreal),len(trainfake1))
    if index==0:
        newtraindata=trainreal*2+trainfake1
    if index==1:
        newtraindata=trainreal*6+trainfake1
    if index==2:
        newtraindata=trainreal+trainfake1
    newvaldata=valreal+valfake1
    print('val',len(valreal),len(valfake1))
    
    data={}
    data['train']=newtraindata
    data['val']=newvaldata
    with open('jsons/data_'+pnames[index]+'.json', 'w') as json_file:
        json.dump(data,json_file)
