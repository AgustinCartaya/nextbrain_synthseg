import os
from tree_sampling import LabelTree


t = LabelTree.new(colors=['GM', 'WM', 'CSF'], contrasts=['T1', 'T2', 'FLAIR', 'PD'], version='synthseg-0.1',
                  description='FreeSurfer/SynthSeg labels. Left and right share a node. '
                              'r_max values are illustrative, not measured (WP1 will estimate them).')
A = t.add_node

# ---- Level 1 ----------------------------------------------------------------------------------
A('background', 'root', labels=[0], null=True)
A('cerebral_wm', 'root', labels=[2, 41], w_mu=[0, 1, 0], wr_mu=4, wr_sigma=0.5)
A('cortex', 'root', labels=[3, 42], w_mu=[1, 0, 0], wr_mu=4, wr_sigma=0.5)
A('csf_spaces', 'root', labels=[24], w_mu=[0, 0, 1], wr_mu=4, wr_sigma=0.5)
A('deep_gm', 'root', w_mu=[0.7, 0.3, 0], wr_mu=6, wr_sigma=1)
A('medial_temporal', 'root', w_mu=[0.95, 0.05, 0], wr_mu=6, wr_sigma=1)
A('cerebellum', 'root', w_mu=[0.6, 0.4, 0], wr_mu=6, wr_sigma=1)
A('brainstem', 'root', labels=[16], w_mu=[0.3, 0.7, 0], wr_mu=6, wr_sigma=1)
A('vessels', 'root', labels=[30, 62], w_mu=[0.3, 0, 0.7], wr_mu=10, wr_sigma=2)
A('extra_cerebral', 'root', w_mu=[0.34, 0.33, 0.33], wr_mu=40, wr_sigma=3,
  description='Non-brain labels of the SynthSeg training maps (skull, scalp, ...). Large jitter = SynthSeg-like.')

# ---- White matter ------------------------------------------------------------------------------
A('wm_lesions', 'cerebral_wm', labels=[25, 57], wr_mu=20, wr_sigma=2, r_max=2.0)
A('optic_chiasm', 'cerebral_wm', labels=[85], wr_mu=5, wr_sigma=0.5, r_max=1.5)

# ---- CSF spaces --------------------------------------------------------------------------------
A('lateral_ventricles', 'csf_spaces', labels=[4, 43], wr_mu=3, wr_sigma=0.5, r_max=5.0)
A('inf_lat_ventricles', 'csf_spaces', labels=[5, 44], wr_mu=3, wr_sigma=0.5, r_max=1.5)
A('third_ventricle', 'csf_spaces', labels=[14], wr_mu=3, wr_sigma=0.5, r_max=2.0)
A('fourth_ventricle', 'csf_spaces', labels=[15], wr_mu=3, wr_sigma=0.5, r_max=2.0)
A('fifth_ventricle', 'csf_spaces', labels=[72], wr_mu=3, wr_sigma=0.5, r_max=1.0)

# ---- Deep grey matter --------------------------------------------------------------------------
A('striatum', 'deep_gm', wr_mu=6, wr_sigma=1, r_max=3.5)
A('caudate', 'striatum', labels=[11, 50], wr_mu=5, wr_sigma=0.5, r_max=2.0)
A('putamen', 'striatum', labels=[12, 51], wr_mu=5, wr_sigma=0.5, r_max=2.0)
A('accumbens', 'striatum', labels=[26, 58], wr_mu=5, wr_sigma=0.5, r_max=1.2)
A('pallidum', 'deep_gm', labels=[13, 52], wr_mu=10, wr_sigma=1, r_max=2.0)
A('thalamus', 'deep_gm', labels=[10, 49], wr_mu=8, wr_sigma=1, r_max=3.5)
A('ventral_dc', 'deep_gm', labels=[28, 60], wr_mu=8, wr_sigma=1, r_max=1.5)
A('caudate_putamen', 'striatum', labels=[136, 137], wr_mu=5, wr_sigma=0.5, r_max=2.0,
  description='FreeSurfer LUT 136/137: Left/Right-Caudate-Putamen')
A('subcortical_gm', 'deep_gm', labels=[163, 164], wr_mu=5, wr_sigma=0.5, r_max=1.0,
  description='FreeSurfer LUT 163/164: Left/Right-Subcortical-Gray-Matter')

# ---- Medial temporal lobe ----------------------------------------------------------------------
A('hippocampus', 'medial_temporal', labels=[17, 53], wr_mu=6, wr_sigma=0.5, r_max=2.0)
A('amygdala', 'medial_temporal', labels=[18, 54], wr_mu=6, wr_sigma=0.5, r_max=2.0)

# ---- Cerebellum --------------------------------------------------------------------------------
A('cerebellar_wm', 'cerebellum', labels=[7, 46], wr_mu=25, wr_sigma=2, r_max=4.0)
A('cerebellar_cortex', 'cerebellum', labels=[8, 47], wr_mu=25, wr_sigma=2, r_max=4.0)

# # ---- Extra-cerebral (always visible, independent-ish like SynthSeg) -----------------------------
# for lab in [502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 530]:
#     A(f'extra_{lab}', 'extra_cerebral', labels=[lab], wr_mu=40, wr_sigma=3, r_max=None)

# # ---- Cortical segmentation -----------------------------
CEREBRAL_CORTEX_LEFT_96 = [1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]
CEREBRAL_CORTEX_RIGHT_96 = [2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
CEREBRAL_CORTEX_96 = CEREBRAL_CORTEX_LEFT_96 + CEREBRAL_CORTEX_RIGHT_96
for (cortex_l, cortex_r) in zip(CEREBRAL_CORTEX_LEFT_96, CEREBRAL_CORTEX_RIGHT_96):
    A(f'cortex_{cortex_l}_{cortex_r}', 'cortex', labels=[cortex_l, cortex_r], wr_mu=15, wr_sigma=0.5, r_max=1.0)



os.makedirs('data/inputs', exist_ok=True)

t.validate()
# t.to_json('data/inputs/synthseg_tree.json')
t.to_json('data/inputs/synthseg_ctx_tree.json')
print(len(t.nodes), 'nodes,', len(t.leaves()), 'leaves')
