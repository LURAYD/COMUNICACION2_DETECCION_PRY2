"""Regenera solo figuras desde matrices guardadas, sin inferencia."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.conventional import CLASSES


if __name__ == '__main__':
    root = Path('results/fase6_holdout_final')
    for split in ('test','test_shift'):
        fig, axes = plt.subplots(2,2,figsize=(12,11))
        fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.91,wspace=.25,hspace=.30)
        for ax,name in zip(axes.flat,('cnn_original','cnn_phase','svm256','svm512')):
            matrix=np.array(json.loads((root/f'confusion_{split}_{name}.json').read_text()))
            norm=matrix/matrix.sum(1,keepdims=True)
            ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            for i in range(6):
                for j in range(6):
                    ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.5 else 'black')
            ax.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,
                   xlabel='Predicción',ylabel='Verdad',title=name)
        fig.suptitle(f'Holdout independiente: {split}',fontsize=16)
        fig.savefig(root/f'confusions_{split}.png',dpi=140)
        plt.close(fig)
