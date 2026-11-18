#!/usr/bin/env bash
# Builds the candidate segmentations for the 5 NextBrain cases:
#   v1          build_3d_segmentation.py              (nearest section)
#   v1_restore  v1 + dev/wp2/fill_holes/manual/restore.py
#   v2          build_3d_segmentation_v2.py
#   v2_restore  v2 + restore.py (extra combination)
# Outputs and logs go to outputs/methods_comparison/. Cases run in parallel.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/outputs/methods_comparison"
RESTORE=/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/manual/restore.py
RES=0.2
mkdir -p "$OUT"/{v1,v1_restore,v2,v2_restore,tables,figures,logs}
for CASE_DIR in "$HERE"/../NextBrain-Case-*/*/; do
  CASE="$(basename "$CASE_DIR")"
  [ -d "$CASE_DIR/labels_blocks" ] || continue
  (
    /usr/bin/time -v python3 "$HERE/build_3d_segmentation.py" --case-dir "$CASE_DIR" --res $RES --out-dir "$OUT/v1" \
        > "$OUT/logs/${CASE}_v1.log" 2>&1 &&
    /usr/bin/time -v python3 "$RESTORE" "$OUT/v1/${CASE}_seg_${RES}mm.nii.gz" "$OUT/v1_restore/${CASE}_seg_${RES}mm_restored.nii.gz" \
        > "$OUT/logs/${CASE}_v1_restore.log" 2>&1
  ) &
  (
    /usr/bin/time -v python3 "$HERE/build_3d_segmentation_v2.py" --case-dir "$CASE_DIR" --res $RES --out-dir "$OUT/v2" \
        > "$OUT/logs/${CASE}_v2.log" 2>&1 &&
    /usr/bin/time -v python3 "$RESTORE" "$OUT/v2/${CASE}_seg_${RES}mm_v2.nii.gz" "$OUT/v2_restore/${CASE}_seg_${RES}mm_v2_restored.nii.gz" \
        > "$OUT/logs/${CASE}_v2_restore.log" 2>&1
  ) &
done
wait
echo "all done"
