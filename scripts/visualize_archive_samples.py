"""Plot cached 3D samples; color only the bead pixels actually targeted in 2D."""
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from review_training_archive import images



def main(review,runs):
    OUT=review.resolve()
    ROWS=json.loads((OUT/'inventory.json').read_text())
    for id in runs:
        run=next(r for r in ROWS if r['id']==id)
        ev=next(e for e in run['evaluations'] if e['path']==run['best_saved_inference'])
        case=min(ev['cases'],key=lambda c:c['index']);cfg=run['config']
        with (Path(run['path'])/'labels.pickle').open('rb') as f:labels=pickle.load(f)
        item=labels[sorted(labels)[case['index']]]
        xyz=np.asarray(item['xyz']).reshape(-1,3)
        dec=images(tifffile.imread(case['prediction']),tifffile.imread(case['target']))
        # Only attach predictions to coordinates when the cached target is identical.
        target=np.asarray(item['target']).squeeze()
        if target.shape!=dec['truth'].shape or not np.array_equal(target>=.5,dec['truth']):
            raise ValueError(id+': cached scene does not match saved inference target')
        ratio=float(cfg.get('ratio_input_output_image_size',1));px=float(cfg['px'])*1e6
        col=(xyz[:,0]//ratio).astype(int);row=(xyz[:,1]//ratio).astype(int)
        zmin,zmax=cfg['z_range_cost_function'];focus=(xyz[:,2]>=zmin)&(xyz[:,2]<=zmax)
        supported=(row>=0)&(row<dec['truth'].shape[0])&(col>=0)&(col<dec['truth'].shape[1])
        focus&=supported
        detected=np.zeros(len(xyz),bool);detected[focus]=dec['pred'][row[focus],col[focus]]
        status=np.where(~focus,'outside_target_depth',np.where(detected,'TP_projected_pixel','FN_projected_pixel'))
        fp=np.argwhere(~dec['truth']&dec['pred'])
        with (OUT/(id+'_scene0_emitters.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['x_um','y_um','z_um','status'])
            w.writerows([*point,label] for point,label in zip(xyz*px,status))
        with (OUT/(id+'_scene0_false_positive_pixels.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['camera_x_pixel','camera_y_pixel']);w.writerows(fp[:,::-1])
        fig=plt.figure(figsize=(15,6))
        for j,zoom in enumerate([False,True]):
            ax=fig.add_subplot(1,2,j+1,projection='3d')
            visible=np.abs(xyz[:,2])<=8 if zoom else np.ones(len(xyz),bool)
            for mask,color,label,size,alpha in [(~focus,'#8493a2','Outside target depth',4,.15),
                    (focus&detected,'#00a957','TP at bead projection',30,1),
                    (focus&~detected,'#f03232','FN at bead projection',40,1)]:
                p=xyz[visible&mask]*px
                ax.scatter(p[:,0],p[:,1],p[:,2],c=color,s=size,alpha=alpha,label=label)
            if len(fp):ax.scatter(fp[:,1]*ratio*px,fp[:,0]*ratio*px,np.zeros(len(fp)),c='#0879e8',marker='x',s=40,label='FP pixel projected to z=0')
            ax.set(xlabel='x / propagation (µm)',ylabel='y / scan (µm)',zlabel='z / sheet thickness (µm)',
                   title='Case 0, full cached 3D sample' if not zoom else 'Case 0, |z| ≤ 8 sample pixels')
            if zoom:ax.set_zlim(-8*px,8*px)
            ax.set_box_aspect((3,1,1.2));ax.view_init(elev=24,azim=-65)
        ax.legend(fontsize=7,loc='upper left');fig.suptitle(id+' | 2D in-focus task: gray beads are not detection failures\nPixel assignment at bead coordinates; no independent 3D localization or pair resolution')
        fig.tight_layout();fig.savefig(OUT/(id+'_sample3d.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        del labels
        print('3D VERIFIED',id,len(xyz),'emitters;',int(focus.sum()),'in-focus beads;',len(fp),'FP pixels',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--review',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);a=p.parse_args();main(a.review,a.runs)
