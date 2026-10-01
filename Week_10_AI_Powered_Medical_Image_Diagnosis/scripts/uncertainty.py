from pathlib import Path
import argparse,numpy as np,torch,torch.nn as nn
from PIL import Image
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/'models'/'densenet121_pneumonia_best.pth'; OUT=ROOT/'outputs'/'uncertainty'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build():
 m=models.densenet121(weights=None); m.classifier=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,1)); return m
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); ap.add_argument('--passes',type=int,default=20); a=ap.parse_args(); m=build().to(DEV); m.load_state_dict(torch.load(MP,map_location=DEV)['model_state_dict']); m.eval(); [x.train() for x in m.modules() if isinstance(x,nn.Dropout)]; tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); x=tf(Image.open(a.image).convert('RGB')).unsqueeze(0).to(DEV); ps=[]
 with torch.no_grad():
  for _ in range(a.passes): ps.append(torch.sigmoid(m(x)).item())
 ps=np.array(ps); mean=ps.mean(); std=ps.std(); label='PNEUMONIA' if mean>=.5 else 'NORMAL'; flag=std>=.10; OUT.mkdir(parents=True,exist_ok=True); path=OUT/(Path(a.image).stem+'_uncertainty.txt'); path.write_text(f'Prediction: {label}\nMean pneumonia probability: {mean:.6f}\nStandard deviation: {std:.6f}\nMonte Carlo passes: {a.passes}\nHigh-uncertainty flag: {flag}\n'); print(f'Prediction: {label}\nMean probability: {mean:.4f}\nStd: {std:.4f}\nHigh-uncertainty flag: {flag}\nSaved: {path}')
if __name__=='__main__': main()
