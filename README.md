# SynthNextBrain

Learning-based, contrast- and resolution-agnostic segmentation of the full NextBrain histological atlas (333 structures) at 0.33 mm, in seconds per scan.

SynthNextBrain replaces NextBrain's per-scan Bayesian inference with a feed-forward network trained SynthSeg-style: images are synthesised on the fly from high-resolution label maps, so no real training images are needed and the model works on any MRI contrast and resolution.


## Key ideas

- **Resolution-aware ontology.** A four-level label tree (tissues → ~40 structures → ~100–120 nuclear groups → 333 leaves), where each node is tagged with the coarsest resolution at which its children are separable per contrast.
- **Mixed-resolution synthesis.** Inputs are generated at 0.25–5 mm (including anisotropic clinical stacks); the target is always the 0.33 mm leaf map. Leaves that are indistinguishable at the sampled resolution share one intensity Gaussian, so the network can't learn boundaries real MRI never shows.
- **Hierarchical label-set loss.** Unresolvable nodes are supervised on the summed probability of their children; resolvable ones at leaf level.
- **Memory-scalable training.** Patch training, class-sampled logits (K ≈ 48 instead of 333) and a coarse-to-fine cascade make 0.33 mm training fit on 40–80 GB GPUs.
- **Uncertainty.** Per-voxel uncertainty flags leaves the input cannot support.
