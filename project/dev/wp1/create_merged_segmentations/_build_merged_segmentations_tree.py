import os
import re
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'wp3', 'simple_tree_generator'))
from tree_sampling import LabelTree


OUTPUT_JSON = os.path.join(HERE, 'data/outputs/l1_nextbrain_tree.json')


# ---- Level 1 (tissue classes): weights over the principal colors [GM, WM, CSF] -------------------
# Illustrative values (same spirit as _build_synthseg_tree.py); WP1 will estimate them.
# L1_PARAMS = {
#     'Background':       dict(null=True),
#     'Cerebral GM':      dict(w_mu=[1, 0, 0], wr_mu=4, wr_sigma=0.5),
#     'Cerebral WM':      dict(w_mu=[0, 1, 0], wr_mu=4, wr_sigma=0.5),
#     'CSF / ventricles': dict(w_mu=[0, 0, 1], wr_mu=4, wr_sigma=0.5),
#     'Deep GM':          dict(w_mu=[0.7, 0.3, 0], wr_mu=6, wr_sigma=1),
#     'Brainstem':        dict(w_mu=[0.3, 0.7, 0], wr_mu=6, wr_sigma=1),
#     'Cerebellar GM':    dict(w_mu=[0.9, 0.1, 0], wr_mu=6, wr_sigma=1),
#     'Cerebellar WM':    dict(w_mu=[0.1, 0.9, 0], wr_mu=6, wr_sigma=1),
#     'Other':            dict(w_mu=[0.34, 0.33, 0.33], wr_mu=10, wr_sigma=2),
# }

# ---- Random jitter of the deeper nodes (illustrative) ---------------------------------------------
t = LabelTree.new(colors=['GM', 'WM', 'CSF'], contrasts=['T1', 'T2', 'FLAIR', 'PD'], version='synthseg-0.1',
                  description='FreeSurfer/SynthSeg labels. Left and right share a node. '
                              'r_max values are illustrative, not measured (WP1 will estimate them).')
A = t.add_node

# ---- Level 1 ----------------------------------------------------------------------------------
A('background', 'root', labels=[0], null=True)
A('brainstem', 'root', labels=[1], w_mu=[0.3, 0.7, 0], wr_mu=6, wr_sigma=1)
A('csf_ventricles', 'root', labels=[2], w_mu=[0, 0, 1], wr_mu=4, wr_sigma=0.5)
A('cerebellar_gm', 'root', labels=[3], w_mu=[0.9, 0.1, 0], wr_mu=6, wr_sigma=1)
A('cerebellar_wm', 'root', labels=[4], w_mu=[0.1, 0.9, 0], wr_mu=6, wr_sigma=1)
A('cerebral_gm', 'root', labels=[5], w_mu=[1, 0, 0], wr_mu=4, wr_sigma=0.5)
A('cerebral_wm', 'root', labels=[6], w_mu=[0, 1, 0], wr_mu=4, wr_sigma=0.5)
A('deep_gm', 'root', labels=[7], w_mu=[0.7, 0.3, 0], wr_mu=6, wr_sigma=1)
A('other', 'root', labels=[8], w_mu=[0.34, 0.33, 0.33], wr_mu=6, wr_sigma=1)

t.to_json(OUTPUT_JSON)
print(len(t.nodes), 'nodes,', len(t.leaves()), 'leaves')
