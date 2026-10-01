from pathlib import Path
import argparse,cv2,numpy as np,torch,torch.nn as nn
from PIL import Image
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/'models'/'densenet121_pneumonia_best.pth'; OUT=ROOT/'outputs'/'gradcam'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build():
 m=models.densenet121(weights=None); m.classifier=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,1)); return m
class CAM:
 def __init__(self,m,layer): self.m=m; self.a=None; self.g=None; layer.register_forward_hook(lambda _,__,o:setattr(self,'a',o)); layer.register_full_backward_hook(lambda _,__,o:setattr(self,'g',o[0]))
 def run(self,x):
  self.m.zero_grad(); z=self.m(x); z[:,0].backward(); w=self.g.mean((2,3),keepdim=True); c=torch.relu((w*self.a).sum(1)).squeeze().detach().cpu().numpy(); c=(c-c.min())/(c.max()-c.min()+1e-8); return c,torch.sigmoid(z).item()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); a=ap.parse_args(); m=build().to(DEV); m.load_state_dict(torch.load(MP,map_location=DEV)['model_state_dict']); m.eval(); tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); x=tf(Image.open(a.image).convert('RGB')).unsqueeze(0).to(DEV); cam,p=CAM(m,m.features.denseblock4.denselayer16.conv2).run(x); im=cv2.resize(cv2.imread(a.image),(224,224)); heat=cv2.applyColorMap(np.uint8(255*cv2.resize(cam,(224,224))),cv2.COLORMAP_JET); out=cv2.addWeighted(im,.55,heat,.45,0); OUT.mkdir(parents=True,exist_ok=True); path=OUT/(Path(a.image).stem+'_gradcam.jpg'); cv2.imwrite(str(path),out); print('Prediction:', 'PNEUMONIA' if p>=.5 else 'NORMAL'); print('Pneumonia probability:',round(p,4)); print('Saved:',path)
if __name__=='__main__': main()
