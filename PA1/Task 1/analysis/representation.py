import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap as umap_module

def plot_umap(feats, VIZ_SPECS, BB_TITLES, STL10_CLASSES, CLASS_COLORS, OUTPUT_DIR):
    for t_name, (src, tgt, lbl_c, lbl_t) in VIZ_SPECS.items():
        lbl_c = np.array(lbl_c) if not isinstance(lbl_c, np.ndarray) else lbl_c
        lbl_t = np.array(lbl_t) if not isinstance(lbl_t, np.ndarray) else lbl_t
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'UMAP: Clean (○) vs {t_name} (×)', fontsize=13, fontweight='bold')

        for col, bb in enumerate(['resnet','vit','clip']):
            ax  = axes[col]
            fc  = feats[f'{bb}_{src}']
            ft  = feats[f'{bb}_{tgt}']

            combined = np.concatenate([fc, ft], axis=0)
            n_c, n_t = len(fc), len(ft)

            reducer = umap_module.UMAP(n_components=2, n_neighbors=15,
                                        min_dist=0.1, random_state=6304,
                                        verbose=False)
            emb      = reducer.fit_transform(combined)
            emb_c    = emb[:n_c]
            emb_t    = emb[n_c:]

            for cls_idx in range(10):
                col_rgb = CLASS_COLORS[cls_idx]
                mc = lbl_c == cls_idx
                mt = lbl_t == cls_idx
                if mc.any():
                    ax.scatter(emb_c[mc, 0], emb_c[mc, 1],
                               c=[col_rgb], marker='o', s=12, alpha=0.55,
                               label=STL10_CLASSES[cls_idx] if col == 0 else '')
                if mt.any():
                    ax.scatter(emb_t[mt, 0], emb_t[mt, 1],
                               c=[col_rgb], marker='x', s=18, alpha=0.85,
                               linewidths=1.3)

            ax.set_title(BB_TITLES[bb], fontsize=11)
            ax.set_xlabel('UMAP-1', fontsize=9)
            ax.set_ylabel('UMAP-2', fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        handles, lbls = axes[0].get_legend_handles_labels()
        handles += [
            Line2D([0],[0], marker='o', color='gray', ls='', ms=7, alpha=0.7, label='Clean'),
            Line2D([0],[0], marker='x', color='gray', ls='', ms=7, markeredgewidth=1.5, label='Transformed'),
        ]
        lbls += ['Clean', 'Transformed']
        fig.legend(handles, lbls, loc='lower center', ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        safe = t_name.lower().replace(' ','_').replace('=','').replace('/','')
        path = os.path.join(OUTPUT_DIR, f'step6_umap_{safe}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")
