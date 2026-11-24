#!/usr/bin/env bash
# Run all simplification scripts in order, with their default arguments.
# Extra arguments (e.g. --jobs 10) are forwarded to every script.
set -euo pipefail

cd "$(dirname "$0")"

SCRIPTS=(
    # nxballen_to_nxbsimplified.py
    # nxballen_to_nxb6.py
    # nxbsimplified_to_nxb16.py
    nxbsimplified_csf_to_nxb18.py
    # nxbsimplified_to_nxbsupersynth.py
)

for s in "${SCRIPTS[@]}"; do
    echo "=== Running $s ==="
    python "$s" "$@"
done

echo "=== All done ==="



# I want to change the generator_double_tree to work in the following way: I should accept a set of
#   labels L, one morphological tree MT = morphological tree, a set of merging scripts SM, a reference to the principal merging script SMP, an
#   IT = intensities tree, and the necessary configuration C. with L it will create the templates as is generator_tree. then
#   it will perfomr the online tissue augmentation with MT also as is generator_tree. then it will use all SM to merge the
#   augmented label with each of these scripts and save it as an specific GT at the output resolution using the same strategie
#   of PV to reduce the size of the labels. then with the resulting label from SMP and the IT the image creation will continue
#   as in generator_tree. In summary, this generator should generate the GT labels at different substructures (one will all,
#   another with 16 (mergedfrom all), another with 30 (merged from all), and the create the intensity images with with the
#   selected level and the given tree for that label.