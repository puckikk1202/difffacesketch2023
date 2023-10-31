import argparse
import torchvision
import torch.nn.functional as F
import torch
import numpy as np


from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="cos",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 3, 4),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(16,16),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        # ---------------  UNET Channel ------------------
        img_channels=11,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 512,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (64, 64), 11, None,
        # ------ inpaint ------
        # model, (64, 64), 6, None,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion

def binarize(input, threshold=0):
    mask = input > threshold
    input[mask] = 1
    input[~mask] = -1
    return input

def get_mask(batch_size, image_size):

    min_ht = image_size // 4
    max_ht = min_ht * 3
    if batch_size != 0:
        masks = torch.zeros(batch_size,1,image_size,image_size)
        for i in range(batch_size):
            mask_w = np.random.randint(min_ht, max_ht)
            mask_h = np.random.randint(min_ht, max_ht)
            px = np.random.randint(0, image_size-mask_w)
            py = np.random.randint(0, image_size-mask_h)
            # pr = np.random.randint(0, 45)
            mask = torch.zeros(1, image_size, image_size)
            mask[:,py:py+mask_h, px:px+mask_w] = 1
            # mask = Rotate(pr)(mask)
            masks[i] = mask
        unknown = masks > 0.5
        masks[unknown] = 1
        masks[~unknown] = -1
        map_mask = binarize(F.interpolate(masks, mode='bilinear', scale_factor=0.25, 
                                      align_corners=True), threshold=-1)
        return masks, map_mask
    else:
        mask_w = np.random.randint(min_ht, max_ht)
        mask_h = np.random.randint(min_ht, max_ht)
        px = np.random.randint(0, image_size-mask_w)
        py = np.random.randint(0, image_size-mask_h)
        # pr = np.random.randint(0, 45)
        mask = torch.zeros(1, image_size, image_size)
        unknown = mask > 0.5
        mask[unknown] = 1
        mask[~unknown] = -1
        mask = mask.unsqueeze(0)
        map_mask = binarize(F.interpolate(mask, mode='bilinear', scale_factor=0.25, 
                                      align_corners=True), threshold=-1)
        map_mask = map_mask.squeeze(0)
        mask = mask.squeeze(0)
        return mask, map_mask

    
    
    # masks = torch.ones(batch_size,1,image_size,image_size)
    # map_mask = binarize(F.interpolate(masks, mode='bilinear', scale_factor=0.25, 
                                    #   align_corners=True), threshold=-1)
    