#!/usr/bin/env bash
# =============================================================================
# End-to-end smoke test (a few minutes, one GPU): 40 scans, a tiny network,
# 8 training iterations, then evaluation and inference. It checks that every
# step runs, NOT that the model is any good. Everything goes to smoke_test_output/, which is
# deleted at the end if all steps pass (keep it with KEEP=1).
#
#   bash launchers/smoke_test.sh            # uses GPU 0; set CUDA_VISIBLE_DEVICES to change
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."            # the project folder
rm -rf smoke_test_output
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CFG=configs/smoke.yaml

echo "== 1. index (40 random scans)"
python 01_build_index.py --config $CFG --limit 40

echo "== 2. small crop of one real test case (label merge + crop, in memory)"
python - <<'PY'
import glob, os
import nibabel as nib, numpy as np
from lib.config import load_config
from lib.labels import LabelMerger, shift_affine
cfg = load_config("configs/smoke.yaml")
src = sorted(glob.glob(os.path.join(cfg["paths"]["test_raw_dir"], "*.nii*")))[0]
img = nib.load(src)
lo = [s // 2 - 80 for s in img.shape]                       # 160^3 block at the centre
raw = np.asarray(img.dataobj[lo[0]:lo[0] + 160, lo[1]:lo[1] + 160, lo[2]:lo[2] + 160])
merged, unknown = LabelMerger(cfg)(raw, extra=cfg["labels"]["test_extra_label_map"])
os.makedirs(cfg["paths"]["test_l1_dir"], exist_ok=True)
nib.save(nib.Nifti1Image(merged, shift_affine(img.affine, lo)),
         os.path.join(cfg["paths"]["test_l1_dir"], "crop_" + os.path.basename(src)))
print("labels", np.unique(merged).tolist(), "| unknown raw labels set to 0:", unknown)
PY

echo "== 3. cache the merged training labels"
python 03_preprocess_cache.py --config $CFG --workers 8

echo "== 4. train (2 epochs x 4 iterations) and resume for one more epoch"
python 04_train.py --config $CFG
python 04_train.py --config $CFG --resume --set train.epochs=3

echo "== 5. evaluation: patches, then one whole volume"
python 06_evaluate.py --config $CFG --mode patch
python 06_evaluate.py --config $CFG --mode volume --save-volumes 1

echo "== 6. inference + QC on the real crop"
python 05_infer.py --config $CFG

if [ "${KEEP:-0}" = "1" ]; then
  echo "== smoke test PASSED: outputs kept in smoke_test_output/"
else
  rm -rf smoke_test_output
  echo "== smoke test PASSED (smoke_test_output/ removed; run with KEEP=1 to keep it)"
fi
