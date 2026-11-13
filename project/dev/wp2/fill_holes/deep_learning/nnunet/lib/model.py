"""
Residual-encoder 3D U-Net with a residual output.

The backbone is nnU-Net's `ResidualEncoderUNet` (package
dynamic_network_architectures, installed with nnunetv2): residual blocks in the
encoder, instance normalisation, LeakyReLU, strided convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationNet(nn.Module):
    """Corrupted label map -> logits of the clean label map.

    Input channels: num_classes one-hot channels of the corrupted labels
                    + 1 "suspicious holes" channel.
    Output:         logits = backbone(input) + alpha * onehot(corrupted)

    The second term makes "keep the input label" the default answer: with a
    backbone output of zero the prediction is the input. The network only has
    to learn the corrections.

    mcfg (section `model` of the config):
      features, blocks   per resolution level (their length is the number of levels;
                         the patch size must be divisible by 2 ** (levels - 1))
      decoder_convs      convolutions per decoder level
      alpha              weight of the input prior
      learn_alpha        train alpha as a scalar parameter
      alpha_skip_holes   do not add the prior on hole-channel voxels
    """

    def __init__(self, num_classes, mcfg):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        n = len(mcfg["features"])
        self.num_classes = int(num_classes)
        self.backbone = ResidualEncoderUNet(
            input_channels=self.num_classes + 1,
            n_stages=n,
            features_per_stage=list(mcfg["features"]),
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=[1] + [2] * (n - 1),
            n_blocks_per_stage=list(mcfg["blocks"]),
            num_classes=self.num_classes,
            n_conv_per_stage_decoder=int(mcfg["decoder_convs"]),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=dict(eps=1e-5, affine=True),
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=dict(inplace=True),
            deep_supervision=False,
        )
        # The backbone always builds one output layer per decoder level (for deep
        # supervision). Only the last one is used here: freeze the others, otherwise
        # DDP stops with "parameters that were not used in producing loss".
        for layer in self.backbone.decoder.seg_layers[:-1]:
            for prm in layer.parameters():
                prm.requires_grad_(False)

        # Zero-initialise the output layer: the untrained network returns exactly
        # alpha * onehot(input), i.e. the identity mapping, and learns corrections from there.
        head = self.backbone.decoder.seg_layers[-1]
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

        alpha = torch.tensor(float(mcfg["alpha"]))
        if mcfg.get("learn_alpha", False):
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.alpha_skip_holes = bool(mcfg.get("alpha_skip_holes", False))

    def forward(self, labels, hole):
        """labels: [B, X, Y, Z] integer corrupted labels; hole: [B, X, Y, Z] 0/1.
        Returns float32 logits [B, num_classes, X, Y, Z]."""
        onehot = F.one_hot(labels.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()
        hole = hole[:, None].float()
        out = self.backbone(torch.cat([onehot, hole], dim=1)).float()
        prior = onehot * (1.0 - hole) if self.alpha_skip_holes else onehot
        return out + self.alpha * prior


def build_model(cfg):
    """Model for a config (uses labels.num_classes and the `model` section)."""
    return RestorationNet(cfg["labels"]["num_classes"], cfg["model"])


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by 04_train.py.

    Returns (model in eval mode on `device`, config stored with the weights, checkpoint dict).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["cfg"], ckpt
