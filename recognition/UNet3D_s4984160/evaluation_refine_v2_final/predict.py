"""Render the selected evaluation predictions without rerunning inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def planes_for(target, geometry):
    z = np.flatnonzero((target==5).any(axis=(1,2)))
    if not len(z):
        z = np.array([0,target.shape[0]//2,target.shape[0]-1])
    chosen = [int(z[0]),int(z[len(z)//2]),int(z[-1])]
    names = ['low array extent','mid-gland','high array extent']
    if geometry['original_axcodes'][2]=='S':
        names = ['inferior GT extent (apex proxy)','mid-gland','superior GT extent (base proxy)']
    result = [('axial',d,f'{name}; D={d}') for d,name in zip(chosen,names)]
    w = geometry['shared_target_w_indices']
    result.append(('coronal',w[len(w)//2],f'coronal shared-band midpoint; W={w[len(w)//2]}'))
    return result


def render(image,target,prediction,meta,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    organ_colors = ListedColormap(['cyan','orange','magenta'])
    error_colors = ListedColormap(['lime','red','dodgerblue'])
    fig,axes = plt.subplots(4,4,figsize=(15,13),constrained_layout=True)
    low,high = np.percentile(image,[1,99])
    for row,(kind,index,label) in enumerate(planes_for(target,meta)):
        if kind=='axial':
            x,y,p = image[index].T,target[index].T,prediction[index].T
            origin = 'upper'
        else:
            x,y,p = image[:,:,index],target[:,:,index],prediction[:,:,index]
            origin = 'lower'
        for col in range(4):
            axes[row,col].imshow(x,cmap='gray',vmin=low,vmax=high,origin=origin)
            axes[row,col].axis('off')
        for col,mask in [(1,y),(2,p)]:
            overlay = np.ma.masked_where(~np.isin(mask,[3,4,5]),mask.astype(int)-3)
            axes[row,col].imshow(overlay,cmap=organ_colors,vmin=0,vmax=2,alpha=.55,origin=origin)
        truth,pred = y==5,p==5
        errors = np.full(truth.shape,-1)
        errors[truth & pred]=0; errors[pred & ~truth]=1; errors[truth & ~pred]=2
        axes[row,3].imshow(np.ma.masked_less(errors,0),cmap=error_colors,vmin=0,vmax=2,alpha=.7,origin=origin)
        for col,title in enumerate(['MRI','Ground truth','Prediction','Prostate TP / FP / FN']):
            axes[row,col].set_title(f'{title}\n{label}',fontsize=9)
    dice = meta['prostate_dice']; distance = meta['prostate_hd95_mm']
    fig.suptitle(f'{meta["case"]} | {meta["split"]} | prostate Dice={dice} | HD95={distance} mm\n'
                 f'Selection: {meta["selection_reason"]}\n'
                 'Organs: bladder cyan, rectum orange, prostate magenta. Errors: TP green, FP red, FN blue.\n'
                 'Resized grid; anatomical extents are GT-derived proxies.',fontsize=11)
    fig.savefig(path,dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    inputs,output = Path(args.input_dir),Path(args.output_dir)
    manifest = json.loads((inputs/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    figures = []
    for row in manifest['cases']:
        with np.load(inputs/row['cache_file'],allow_pickle=False) as data:
            meta = json.loads(str(data['metadata'].item()))
            name = row['case'].replace('.nii.gz','')+'.png'
            render(data['image'],data['target'],data['prediction'],meta,output/name)
        figures.append(dict(row,figure=name))
        print(f'Saved {output/name}',flush=True)
    (output/'manifest.json').write_text(json.dumps(dict(
        checkpoint=manifest['checkpoint'],checkpoint_sha256=manifest['checkpoint_sha256'],
        split=manifest['split'],figures=figures),indent=2),encoding='utf-8')
    (output/'plots_complete.txt').write_text('Prediction figures completed successfully.\n')


if __name__ == '__main__':
    main()
