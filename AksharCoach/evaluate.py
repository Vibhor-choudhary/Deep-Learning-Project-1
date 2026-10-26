"""Held-out clean and distorted evaluation; no test example is used for tuning."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models import build_model
from preprocessing import base_transform

ROOT=Path(__file__).resolve().parent
def corrupt_transform():
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((32,32)), transforms.RandomAffine(8,translate=(.06,.06)), transforms.GaussianBlur(3, sigma=(.1,.6)), transforms.ToTensor(), transforms.Lambda(lambda x: (x+torch.randn_like(x)*.03).clamp(0,1)), transforms.Normalize((.5,),(.5,))])
def infer(model, loader, device):
    model.eval(); pred=[]; actual=[]; conf=[]
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device)); p=out.softmax(1).cpu(); pred.extend(p.argmax(1).tolist()); conf.extend(p.max(1).values.tolist()); actual.extend(y.tolist())
    return np.array(actual),np.array(pred),np.array(conf)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plots-only',action='store_true'); args=parser.parse_args()
    data=ROOT/'data'/'DevanagariHandwrittenCharacterDataset'/'Test'; device='cuda' if torch.cuda.is_available() else 'cpu'
    clean=ImageFolder(data,transform=base_transform()); distorted=ImageFolder(data,transform=corrupt_transform()); loaders={'clean':DataLoader(clean,batch_size=128,shuffle=False),'distorted':DataLoader(distorted,batch_size=128,shuffle=False)}
    hist=json.loads((ROOT/'outputs'/'training_history.json').read_text()); summary={'dataset':{'test_images':len(clean),'classes':len(clean.classes)},'models':{}}
    if args.plots_only:
        ckpt=torch.load(ROOT/'models'/'best_cnn.pt',map_location=device,weights_only=False); model=build_model('cnn',len(clean.classes)).to(device); model.load_state_dict(ckpt['state_dict'])
        y,p,_=infer(model,loaders['clean'],device); cm_for_cnn=confusion_matrix(y,p)
        plt.figure(figsize=(10,8)); plt.imshow(cm_for_cnn,cmap='Blues'); plt.title('CNN confusion matrix (held-out clean test set)'); plt.xlabel('Predicted class index'); plt.ylabel('True class index'); plt.colorbar(); plt.tight_layout(); plt.savefig(ROOT/'outputs'/'confusion_matrix.png',dpi=160); plt.close()
        plt.figure(figsize=(9,4))
        for name,colour in [('mlp','tab:orange'),('cnn','tab:blue')]:
            h=hist['models'][name]['history']; plt.plot([x['epoch'] for x in h],[x['train_accuracy'] for x in h],color=colour,linestyle='--',label=f'{name} train'); plt.plot([x['epoch'] for x in h],[x['val_accuracy'] for x in h],color=colour,label=f'{name} validation')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training and validation accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'outputs'/'training_curves.png',dpi=160); plt.close(); print('Saved figures'); return
    all_rows=[]; cm_for_cnn=None
    for name in ['mlp','cnn']:
        ckpt=torch.load(ROOT/'models'/f'best_{name}.pt',map_location=device,weights_only=False); model=build_model(name,len(clean.classes)).to(device); model.load_state_dict(ckpt['state_dict']); threshold=hist['models'][name]['threshold']; model_result={'threshold':threshold}
        for condition,loader in loaders.items():
            y,p,c=infer(model,loader,device); precision,recall,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0); accepted=c>=threshold
            result={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),'accepted_percentage':float(accepted.mean()),'accepted_accuracy':float(accuracy_score(y[accepted],p[accepted])) if accepted.any() else None}
            model_result[condition]=result; all_rows.append({'model':name,'condition':condition,**result})
            if name=='cnn' and condition=='clean':
                cm_for_cnn=confusion_matrix(y,p); report=classification_report(y,p,target_names=clean.classes,output_dict=True,zero_division=0)
                with (ROOT/'outputs'/'classification_report.csv').open('w',newline='') as f:
                    writer=csv.writer(f); writer.writerow(['class','precision','recall','f1-score','support']); [writer.writerow([k,v.get('precision',''),v.get('recall',''),v.get('f1-score',''),v.get('support','')]) for k,v in report.items() if isinstance(v,dict)]
        summary['models'][name]=model_result
    with (ROOT/'outputs'/'robustness_results.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    (ROOT/'outputs'/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.figure(figsize=(10,8)); plt.imshow(cm_for_cnn,cmap='Blues'); plt.title('CNN confusion matrix (held-out clean test set)'); plt.xlabel('Predicted class index'); plt.ylabel('True class index'); plt.colorbar(); plt.tight_layout(); plt.savefig(ROOT/'outputs'/'confusion_matrix.png',dpi=160); plt.close()
    plt.figure(figsize=(9,4));
    for name,colour in [('mlp','tab:orange'),('cnn','tab:blue')]:
        h=hist['models'][name]['history']; plt.plot([x['epoch'] for x in h],[x['train_accuracy'] for x in h],color=colour,linestyle='--',label=f'{name} train'); plt.plot([x['epoch'] for x in h],[x['val_accuracy'] for x in h],color=colour,label=f'{name} validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training and validation accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'outputs'/'training_curves.png',dpi=160); plt.close()
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
