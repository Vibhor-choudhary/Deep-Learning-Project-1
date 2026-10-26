"""Train MLP and CNN fairly on the same balanced, limited-data subset."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from models import build_model, parameter_count
from preprocessing import base_transform, train_transform, save_label_mapping

ROOT = Path(__file__).resolve().parent

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def balanced_indices(targets, per_class, seed):
    rng = random.Random(seed); chosen = []
    for cls in sorted(set(targets)):
        members = [i for i, y in enumerate(targets) if y == cls]
        rng.shuffle(members); chosen.extend(members[:min(per_class, len(members))])
    rng.shuffle(chosen); return chosen

def accuracy_loss(model, loader, device, criterion, collect=False):
    model.eval(); correct = total = 0; loss_sum = 0; probs=[]; labels=[]
    with torch.no_grad():
        for x, y in loader:
            x,y=x.to(device),y.to(device); out=model(x); loss=criterion(out,y)
            loss_sum += loss.item()*len(y); total += len(y); correct += (out.argmax(1)==y).sum().item()
            if collect: probs.append(out.softmax(1).cpu()); labels.append(y.cpu())
    result=(loss_sum/total, correct/total)
    return (*result, torch.cat(probs), torch.cat(labels)) if collect else result

def choose_threshold(probs, labels):
    conf, pred=probs.max(1); correct=pred.eq(labels)
    # Validation-only: retain the widest useful coverage whose accepted accuracy
    # is at least 90%. This is more practical than rejecting most good answers.
    candidates=[]
    for t in np.arange(0.20, 0.96, 0.05):
        mask=conf>=t
        accepted_accuracy=correct[mask].float().mean().item() if mask.any() else 0.0
        if accepted_accuracy >= .90: candidates.append((mask.float().mean().item(), -float(t)))
    return -max(candidates, default=[1.0, 0.0])[1]

def run_model(name, train_loader, val_loader, args, device):
    model=build_model(name).to(device); optimiser=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4); criterion=nn.CrossEntropyLoss()
    history=[]; best=-1; patience=0; checkpoint=ROOT/'models'/f'best_{name}.pt'
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter(); model.train(); seen=correct=0; loss_sum=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimiser.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimiser.step()
            seen+=len(y); loss_sum+=loss.item()*len(y); correct+=(out.argmax(1)==y).sum().item()
        vloss,vacc=accuracy_loss(model,val_loader,device,criterion)
        row={'epoch':epoch,'train_loss':loss_sum/seen,'train_accuracy':correct/seen,'val_loss':vloss,'val_accuracy':vacc,'seconds':time.perf_counter()-started}; history.append(row)
        print(f"{name} epoch {epoch}/{args.epochs}: train_acc={row['train_accuracy']:.3f} val_acc={vacc:.3f} ({row['seconds']:.1f}s)")
        if vacc > best:
            best=vacc; patience=0; torch.save({'model':name,'state_dict':model.state_dict(),'classes':args.classes,'config':vars(args)},checkpoint)
        else:
            patience+=1
            if patience>=args.patience: break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'])
    _,_,probs,labels=accuracy_loss(model,val_loader,device,criterion,collect=True)
    return {'parameters':parameter_count(model),'best_val_accuracy':best,'threshold':choose_threshold(probs,labels),'history':history}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(ROOT/'data'/'DevanagariHandwrittenCharacterDataset')); p.add_argument('--per-class',type=int,default=25); p.add_argument('--epochs',type=int,default=4); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--patience',type=int,default=3); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    seed_everything(args.seed); data=Path(args.data); train_dir=data/'Train'
    if not train_dir.exists(): raise FileNotFoundError(f"Dataset missing: {train_dir}; see data/README.md")
    plain=ImageFolder(train_dir, transform=base_transform()); args.classes=len(plain.classes)
    if args.classes != 46: raise ValueError(f"Expected 46 classes, found {args.classes}")
    indices=balanced_indices(plain.targets,args.per_class,args.seed); split=max(1,int(.15*len(indices))); val_idx,train_idx=indices[:split],indices[split:]
    augmented=ImageFolder(train_dir, transform=train_transform())
    train_loader=DataLoader(Subset(augmented,train_idx),batch_size=args.batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(Subset(plain,val_idx),batch_size=args.batch_size,shuffle=False,num_workers=0)
    device='cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device={device}; train={len(train_idx)} validation={len(val_idx)} classes={args.classes}')
    (ROOT/'models').mkdir(exist_ok=True); save_label_mapping(plain.classes,ROOT/'models'/'labels.json')
    results={'configuration':vars(args),'device':device,'models':{}}
    for name in ['mlp','cnn']: results['models'][name]=run_model(name,train_loader,val_loader,args,device)
    (ROOT/'outputs'/'training_history.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Saved checkpoints and outputs/training_history.json')
if __name__=='__main__': main()
