#!/usr/bin/env python3
"""Local Gradio demo. Run `python3 app.py` or `./app.py` after training and evaluation."""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from models import build_model
from preprocessing import upload_to_tensor

ROOT=Path(__file__).resolve().parent
def load():
    labels=json.loads((ROOT/'models'/'labels.json').read_text()); hist=json.loads((ROOT/'outputs'/'training_history.json').read_text()); ckpt=torch.load(ROOT/'models'/'best_cnn.pt',map_location='cpu',weights_only=False); model=build_model('cnn',len(labels)); model.load_state_dict(ckpt['state_dict']); model.eval(); return model,labels,hist['models']['cnn']['threshold']
MODEL,LABELS,THRESHOLD=load()
def predict(image):
    if image is None: return None, 'Please upload one character image.', []
    raw_parts=ink_component_count(image)
    if raw_parts > 12:
        return image, (f'This image contains approximately {raw_parts} separate ink regions, so it is not a one-character input. '
                       'Open “Analyse separate handwritten characters” for a plain sheet of spaced handwriting. '
                       'A printed reference chart/screenshot is outside this handwriting model’s supported input.'), []
    x=upload_to_tensor(image)
    with torch.no_grad(): p=MODEL(x).softmax(1)[0]
    values,indices=torch.topk(p,3); rows=[]
    for score,idx in zip(values.tolist(),indices.tolist()):
        item=LABELS[idx]; rows.append([item['glyph'],item['folder'],round(score*100,2)])
    accepted=values[0].item()>=THRESHOLD
    status=(f"Prediction accepted (model score {values[0]:.1%}, validation-selected threshold {THRESHOLD:.0%})." if accepted else f"Uncertain: top model score {values[0]:.1%} is below the validation-selected threshold {THRESHOLD:.0%}. Please upload a clearer, tightly cropped character or redraw it.")
    return image,status,rows


def ink_component_count(image):
    """A conservative guardrail for the one-character workflow."""
    arr=np.asarray(ImageOps.grayscale(image))
    if arr.mean() < 127: arr=255-arr
    labels,count=ndimage.label(arr < 200, structure=np.ones((3,3),dtype=int))
    meaningful=0
    for value in range(1,count+1):
        ys,xs=np.where(labels==value)
        if len(xs) >= 18 and (xs.max()-xs.min()+1) >= 3 and (ys.max()-ys.min()+1) >= 3:
            meaningful+=1
    return meaningful


def predict_crop(crop):
    """Classify one already-isolated crop and return the best prediction."""
    x=upload_to_tensor(crop)
    with torch.no_grad(): scores=MODEL(x).softmax(1)[0]
    score,index=scores.max(0); item=LABELS[index.item()]
    return item['glyph'], item['folder'], score.item(), score.item() >= THRESHOLD


def segment_characters(image):
    """Find separately spaced, dark handwritten characters on a light page.

    This deliberately does not claim to read printed charts, connected words, or
    table layouts. It is for a page of individually spaced handwritten symbols.
    """
    gray=ImageOps.grayscale(image)
    arr=np.asarray(gray)
    if arr.mean() < 127: arr=255-arr
    ink=arr < 200
    labels,count=ndimage.label(ink, structure=np.ones((3,3),dtype=int))
    candidates=[]; total=arr.size
    for label in range(1,count+1):
        ys,xs=np.where(labels==label)
        if not len(xs): continue
        x0,y0,x1,y1=xs.min(),ys.min(),xs.max()+1,ys.max()+1
        area=len(xs); width,height=x1-x0,y1-y0
        # Reject tiny dots/noise and extreme divider lines
        if area < 18 or width < 4 or height < 4 or area > total*.22: continue
        if width > height*8 or height > width*8: continue
        candidates.append({'box': (x0, y0, x1, y1), 'area': area, 'w': width, 'h': height})
    
    if not candidates: return []
    # Filter out dashed grid lines and small border noise relative to main character bodies
    max_area = max(c['area'] for c in candidates)
    max_h = max(c['h'] for c in candidates)
    max_w = max(c['w'] for c in candidates)
    
    pad = 5; boxes = []
    for c in candidates:
        if max_area > 300 and (c['area'] < max_area * 0.05 or c['h'] < max_h * 0.22 or c['w'] < max_w * 0.15):
            continue
        x0,y0,x1,y1 = c['box']
        boxes.append((max(0,x0-pad),max(0,y0-pad),min(image.width,x1+pad),min(image.height,y1+pad)))

    # Reading order: cluster nearby y coordinates into visual rows.
    boxes.sort(key=lambda b:(b[1],b[0])); rows=[]
    for box in boxes:
        centre=(box[1]+box[3])//2
        row=next((r for r in rows if abs(r[0]-centre)<(max_h*0.3 if 'max_h' in locals() and max_h > 30 else 25)),None)
        if row is None: rows.append([centre,[box]])
        else: row[1].append(box)
    return [box for _,row in rows for box in sorted(row,key=lambda b:b[0])]


def analyze_page(image):
    if image is None: return None, 'Please upload a page containing spaced, isolated handwritten characters.', []
    boxes=segment_characters(image)
    if not boxes: return image, 'No separate handwritten-character regions were found. Use a plain, light-background image with characters spaced apart.', []
    annotated=image.convert('RGB').copy(); draw=ImageDraw.Draw(annotated); rows=[]
    for number,box in enumerate(boxes,1):
        glyph,folder,score,accepted=predict_crop(image.crop(box))
        colour=(29,140,65) if accepted else (205,75,45)
        draw.rectangle(box,outline=colour,width=3); draw.text((box[0],max(0,box[1]-15)),str(number),fill=colour)
        rows.append([number,glyph,folder,round(score*100,2),'Accepted' if accepted else 'Uncertain'])
    message=(f'Found and analysed {len(boxes)} separate region(s). Green boxes meet the validation threshold; orange boxes need review. '
             'This page mode is for separately spaced handwritten symbols, not printed alphabet charts or connected words.')
    return annotated,message,rows


def batch_demo_path():
    """Create a presentable 12-character sheet from held-out handwritten data."""
    path=ROOT/'outputs'/'batch_demo.png'
    if path.exists(): return str(path)
    files=[]
    for folder in sorted((ROOT/'data'/'DevanagariHandwrittenCharacterDataset'/'Test').iterdir()):
        found=next(folder.glob('*.png'),None)
        if found: files.append(found)
        if len(files)==12: break
    sheet=Image.new('L',(4*90,3*90),255)
    for n,file in enumerate(files):
        tile=ImageOps.invert(Image.open(file).convert('L')).resize((52,52))
        x=(n%4)*90+19; y=(n//4)*90+19; sheet.paste(tile,(x,y))
    path.parent.mkdir(exist_ok=True); sheet.save(path); return str(path)
def main():
    try: import gradio as gr
    except ImportError as exc: raise SystemExit('Install requirements first: python -m pip install -r requirements.txt') from exc
    metrics=json.loads((ROOT/'outputs'/'metrics.json').read_text()); robustness=pd.read_csv(ROOT/'outputs'/'robustness_results.csv')
    examples=[str(p) for p in list((ROOT/'data'/'DevanagariHandwrittenCharacterDataset'/'Test').rglob('*.png'))[:3]]
    batch_example=batch_demo_path()
    with gr.Blocks(title='AksharCoach') as app:
        gr.Markdown('# AksharCoach\nOffline recognition of **one isolated handwritten Devanagari character or digit**. It does not read Hindi words, sentences, or paragraphs.')
        with gr.Tab('Analyse separate handwritten characters'):
            gr.Markdown('### Batch mode — use this for more than one character\nUpload a plain image with **separately spaced handwritten characters**. The app finds each region, classifies each one, and marks uncertain results. It cannot reliably interpret a printed alphabet chart, web-page screenshot, connected handwritten word, or paragraph because the model was trained only on isolated handwriting.')
            with gr.Row(): page=gr.Image(type='pil',label='Multi-character handwriting page'); annotated=gr.Image(type='pil',label='Detected characters and decisions')
            page_status=gr.Textbox(label='Batch-analysis result'); page_table=gr.Dataframe(headers=['#','Glyph','Dataset class','Model score (%)','Decision'],label='Every detected character')
            gr.Button('Analyse all detected handwritten characters', variant='primary').click(analyze_page,page,[annotated,page_status,page_table])
            gr.Examples(examples=[batch_example],inputs=page,label='Try this bundled 12-character handwritten batch')
        with gr.Tab('Recognize one character only'):
            gr.Markdown('### Single-character mode\nUpload one tightly cropped handwritten character. If the upload contains many characters, the app will stop and direct you to batch mode instead of returning a misleading answer.')
            with gr.Row(): image=gr.Image(type='pil',label='Upload one PNG/JPG character'); processed=gr.Image(type='pil',label='Input image')
            status=gr.Textbox(label='Confidence-aware decision'); table=gr.Dataframe(headers=['Glyph','Dataset class','Model score (%)'],label='Top 3 predictions')
            gr.Button('Recognize').click(predict,image,[processed,status,table])
            gr.Examples(examples=examples, inputs=image, label='Bundled held-out examples')
        with gr.Tab('Model evaluation'):
            gr.JSON(metrics,label='Actual held-out metrics'); gr.Dataframe(robustness,label='Clean vs distorted evaluation'); gr.Image(str(ROOT/'outputs'/'training_curves.png')); gr.Image(str(ROOT/'outputs'/'confusion_matrix.png'))
        with gr.Tab('About this project'):
            gr.Markdown('**Question:** Does a compact CNN remain more robust than a flattened-image MLP with limited training data?\n\nThe UCI Devanagari Handwritten Character Dataset (CC BY 4.0) supplies 46 classes of 32×32 grayscale images. The CNN retains spatial structure through convolution and pooling; the MLP uses 1,024 flattened pixels. Scores are softmax model scores, not calibrated probabilities.')
    # Loopback only; choose a different local port with AKSHARCOACH_PORT if needed.
    app.launch(server_name='127.0.0.1', server_port=int(os.getenv('AKSHARCOACH_PORT', '7860')), share=False)
if __name__=='__main__': main()
