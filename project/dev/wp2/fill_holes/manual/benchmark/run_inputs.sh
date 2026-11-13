#!/usr/bin/env bash
# Run restore.py and baseline_naive.py on every case of the complete-label and L1 inputs.
#
#   bash run_inputs.sh                 # all 20 runs, 10 in parallel
#   JOBS=4 bash run_inputs.sh          # fewer parallel runs (each restore needs ~20-30 GB RAM)
#   SKIP_EXISTING=1 bash run_inputs.sh # do not redo outputs that already exist
#
# Output (same file names as the inputs):
#   restored/{complete,L1}/case-N.nii.gz  (+ case-N_steps.nii.gz, case-N_report.json)
#   naive/{complete,L1}/case-N.nii.gz     (+ case-N_mask.nii.gz: voxels filled)
#   logs/<method>_<labels>_case-N.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUAL="$(dirname "$HERE")"
DATA="$(dirname "$MANUAL")/data"
IN_COMPLETE="${IN_COMPLETE:-$DATA/inputs/recon_holes_0.2mm}"
IN_L1="${IN_L1:-$DATA/inputs/recon_holes_0.2mm_L1}"
OUT="${OUT:-$DATA/outputs/restored_manual}"
JOBS="${JOBS:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PY="${PYTHON:-python}"

mkdir -p "$OUT"/{restored,naive}/{complete,L1} "$OUT/logs"

# one line per run: method labels input
jobs_list() {
  for labels in complete L1; do
    if [ "$labels" = complete ]; then dir="$IN_COMPLETE"; else dir="$IN_L1"; fi
    for f in "$dir"/case-*.nii.gz; do
      for method in restored naive; do
        echo "$method $labels $f"
      done
    done
  done
}

run_one() {
  method=$1; labels=$2; in=$3
  name=$(basename "$in")
  out="$OUT/$method/$labels/$name"
  log="$OUT/logs/${method}_${labels}_${name%.nii.gz}.log"
  if [ "$SKIP_EXISTING" = 1 ] && [ -f "$out" ]; then
    echo "skip   $method/$labels/$name (exists)"; return 0
  fi
  if [ "$method" = restored ]; then
    flag=""; [ "$labels" = L1 ] && flag="--input-is-l1"
    cmd=("$PY" "$MANUAL/restore.py" "$in" "$out" $flag)
  # else
  #   cmd=("$PY" "$MANUAL/baseline_naive.py" "$in" "$out" --save-mask)
  fi
  echo "start  $method/$labels/$name"
  if "${cmd[@]}" > "$log" 2>&1; then
    echo "done   $method/$labels/$name"
  else
    echo "FAILED $method/$labels/$name (see $log)"; return 1
  fi
}
export -f run_one
export OUT MANUAL PY SKIP_EXISTING

echo "complete labels: $IN_COMPLETE"
echo "L1 labels:       $IN_L1"
echo "output:          $OUT  ($JOBS parallel runs)"
jobs_list | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
echo "all runs finished"
