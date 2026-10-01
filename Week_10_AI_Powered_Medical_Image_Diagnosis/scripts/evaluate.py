from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'chest_xray'; MP=ROOT/'models'/'densenet121_pneumonia_best.pth'; OUT=ROOT/'outputs'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build():
 m=models.densenet121(weights=None); m.classifier=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,1)); return m
def main():
 if not MP.exists(): raise FileNotFoundError('Train first: py -3.12 scripts/train.py')
 tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); ds=datasets.ImageFolder(DATA/'test',transform=tf); dl=DataLoader(ds,batch_size=32)
 m=build().to(DEV); m.load_state_dict(torch.load(MP,map_location=DEV)['model_state_dict']); m.eval(); p=[]; y=[]
 with torch.no_grad():
  for x,t in dl: p.extend(torch.sigmoid(m(x.to(DEV))).cpu().numpy().ravel()); y.extend(t.numpy())
 p=np.array(p); y=np.array(y); pred=(p>=.5).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); sens=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp) if tn+fp else 0; auc=roc_auc_score(y,p)
 print(f'ROC-AUC: {auc:.4f}\nSensitivity: {sens:.4f}\nSpecificity: {spec:.4f}\nAccuracy: {(pred==y).mean():.4f}'); print(cm); print(classification_report(y,pred,target_names=['NORMAL','PNEUMONIA'],zero_division=0))
 (OUT/'evaluation_metrics.txt').write_text(f'ROC-AUC: {auc:.6f}\nSensitivity: {sens:.6f}\nSpecificity: {spec:.6f}\nAccuracy: {(pred==y).mean():.6f}\nTN={tn}, FP={fp}, FN={fn}, TP={tp}\n')
 plt.figure(); plt.imshow(cm); plt.xticks([0,1],['NORMAL','PNEUMONIA']); plt.yticks([0,1],['NORMAL','PNEUMONIA']); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix');
 for i in range(2):
  for j in range(2): plt.text(j,i,cm[i,j],ha='center',va='center')
 plt.tight_layout(); plt.savefig(OUT/'confusion_matrix.png'); plt.close(); fpr,tpr,_=roc_curve(y,p); plt.figure(); plt.plot(fpr,tpr,label=f'AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'roc_curve.png'); plt.close()
if __name__=='__main__': main()
