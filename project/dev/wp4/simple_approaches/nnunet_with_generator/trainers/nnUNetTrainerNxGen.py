"""nnU-Net trainer whose training batches are generated on the fly by nxgen.

Found by nnU-Net through $nnUNet_extTrainer (set in env.sh):

    nnUNetv2_train 101 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerNxGen

Only the data source changes. Network, loss, optimiser, schedule, checkpoints, per-epoch
validation and final validation are nnUNetTrainer's:

- training: `nxgen_batches.GeneratorBatches` instead of nnUNetDataLoader + augmentations
  (every iteration sees new synthetic images; nnU-Net's spatial/intensity augmentations
  are not applied, the generator does its own; only mirroring is kept);
- validation: the standard nnU-Net loader on the fixed preprocessed validation cases of
  this fold (synthetic images of the held-out template, prepare_dataset.py).

Environment (all optional):
    NXGEN_DEVICE          generator device in this process (default: the training device)
    NXGEN_SEED            run seed of the training stream (default 0)
    NXGEN_PREFETCH        batches generated ahead in a background thread (default 2)
    NXGEN_NUM_EPOCHS      default 1000
    NXGEN_ITERS_PER_EPOCH default 250
"""
import os
import shutil
import sys

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from torch import distributed as dist

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

HERE = os.path.dirname(os.path.realpath(__file__))
NXGEN_ROOT = os.environ.get('NXGEN_ROOT', os.path.normpath(join(HERE, '../../../../wp3/generator_v2_modular')))
# appended even if already there: nnU-Net's trainer lookup puts HERE on sys.path only while it
# imports this module, then removes its (first) entry
sys.path += [HERE, NXGEN_ROOT]


class nnUNetTrainerNxGen(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get('NXGEN_NUM_EPOCHS', self.num_epochs))
        self.num_iterations_per_epoch = int(os.environ.get('NXGEN_ITERS_PER_EPOCH', self.num_iterations_per_epoch))
        self.nxgen_seed = int(os.environ.get('NXGEN_SEED', 0))
        self.nxgen_prefetch = int(os.environ.get('NXGEN_PREFETCH', 2))
        self.nxgen_device = os.environ.get('NXGEN_DEVICE')

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        _, _, _, mirror_axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # validation: unchanged nnU-Net pipeline on the fixed validation cases
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        _, dataset_val = self.get_tr_and_val_datasets()
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        n_proc = get_allowed_n_proc_DA()
        if n_proc == 0:
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val, transform=None,
                                                      num_processes=max(1, n_proc // 2),
                                                      num_cached=max(3, n_proc // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_val)

        # training: nxgen. Resuming at epoch e continues the stream where it stopped.
        from nxgen_batches import build_from_plans
        world_size = dist.get_world_size() if self.is_ddp else 1
        gen_train = build_from_plans(
            self.preprocessed_dataset_folder_base, self.plans_manager.plans, self.configuration_name, self.fold,
            self.batch_size, deep_supervision_scales, mirror_axes, self.device, run_seed=self.nxgen_seed,
            start=self.current_epoch * self.num_iterations_per_epoch, rank=self.local_rank,
            world_size=world_size, prefetch=self.nxgen_prefetch, gen_device=self.nxgen_device,
            log=self.print_to_log_file)
        return gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        # generator settings next to the plans, for the record
        if self.local_rank == 0:
            for f in ('nxgen.json', 'nxgen_config.json', 'label_space.json', 'tree.json'):
                src = join(self.preprocessed_dataset_folder_base, f)
                if os.path.isfile(src):
                    shutil.copy(src, join(self.output_folder, f))
            save_json(dict(seed=self.nxgen_seed, prefetch=self.nxgen_prefetch, device=self.nxgen_device,
                           num_epochs=self.num_epochs, iterations_per_epoch=self.num_iterations_per_epoch,
                           fold=self.fold, nxgen_root=NXGEN_ROOT),
                      join(self.output_folder, 'nxgen_runtime.json'))

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        g = self.dataloader_train
        self.print_to_log_file(f'nxgen: {g.gen_s:.1f} s generating, {g.wait_s:.1f} s waited by the training loop')
        g.gen_s = g.wait_s = 0.0

    def on_train_end(self):
        if self.dataloader_train is not None:
            self.dataloader_train.close()
        super().on_train_end()
