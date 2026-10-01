from pathlib import Path
import copy, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'chest_xray'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MODEL_OUT=ROOT/'models'/'densenet121_pneumonia_best.pth'; DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=15; BATCH=32; LR=1e-4
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def model():
    m=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for p in m.features.parameters(): p.requires_grad=False
    for p in m.features.denseblock4.parameters(): p.requires_grad=True
    m.classifier=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,1))
    return m

def main():
    train=DATA/'train'; val=DATA/'val' if (DATA/'val').exists() else DATA/'validation'
    if not train.exists() or not val.exists(): raise FileNotFoundError('Put the Kaggle dataset under data/chest_xray with train and val/validation folders.')
    tr_tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    va_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tr=datasets.ImageFolder(train,transform=tr_tf); va=datasets.ImageFolder(val,transform=va_tf)
    counts=np.bincount(tr.targets,minlength=2); total=counts.sum(); cw=total/(2*np.maximum(counts,1)); sw=[cw[y] for y in tr.targets]
    sampler=WeightedRandomSampler(torch.tensor(sw,dtype=torch.double),len(sw),replacement=True)
    tl=DataLoader(tr,batch_size=BATCH,sampler=sampler); vl=DataLoader(va,batch_size=BATCH,shuffle=False)
    m=model().to(DEVICE); pos_weight=torch.tensor([cw[1]/cw[0]],device=DEVICE); loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,m.parameters()),lr=LR); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2)
    scaler=torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available()); best=-1; hist=[]
    print('Classes:',tr.class_to_idx,'Train:',len(tr),'Validation:',len(va),'Device:',DEVICE)
    for ep in range(1,EPOCHS+1):
        m.train(); s=0
        for x,y in tl:
            x=x.to(DEVICE); y=y.float().to(DEVICE).unsqueeze(1); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda',enabled=torch.cuda.is_available()): z=m(x); loss=loss_fn(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); s+=loss.item()*len(x)
        trloss=s/len(tr); m.eval(); s=0; pr=[]; lab=[]
        with torch.no_grad():
            for x,y in vl:
                x=x.to(DEVICE); yf=y.float().to(DEVICE).unsqueeze(1); z=m(x); s+=loss_fn(z,yf).item()*len(x); pr.extend(torch.sigmoid(z).cpu().numpy().ravel()); lab.extend(y.numpy())
        vloss=s/len(va); auc=roc_auc_score(lab,pr); sched.step(vloss); hist.append([ep,trloss,vloss,auc])
        print(f'Epoch {ep:02d}/{EPOCHS} | train_loss={trloss:.4f} | val_loss={vloss:.4f} | val_auc={auc:.4f}')
        if auc>best:
            best=auc; torch.save({'model_state_dict':copy.deepcopy(m.state_dict()),'class_to_idx':tr.class_to_idx},MODEL_OUT)
    with (OUT/'training_history.csv').open('w',newline='') as f: csv.writer(f).writerows([['epoch','train_loss','val_loss','val_auc'],*hist])
    a=np.array(hist); plt.figure(); plt.plot(a[:,0],a[:,1],label='Train Loss'); plt.plot(a[:,0],a[:,2],label='Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'training_curves.png'); plt.close()
    print('Best validation AUC:',best); print('Saved:',MODEL_OUT)
if __name__=='__main__': main()
