"""
lib — restoration of corrupted NextBrain label maps with a 3D residual U-Net.

Modules
-------
config      YAML loading (`_base_` inheritance, `--set` overrides, path resolution)
external    imports of the two existing scripts used as libraries
            (corrupt_nifti.py and merge_labels.py); neither is modified
labels      label merging (L1 ...), one-hot linear upsampling, cached volume loading
corruption  random corruption arguments + the "suspicious holes" input channel
data        scan index, subject split, sampling weights, patch dataset
model       residual-encoder U-Net with residual output
losses      weighted cross-entropy + soft Dice, and the evaluation metrics
sliding     sliding-window restoration of a whole volume
"""
