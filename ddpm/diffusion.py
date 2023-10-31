import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.parameterization = "eps"
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.v_posterior = 0.

        self.logvar = torch.full(fill_value=0., size=(self.num_timesteps,))

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

        ## SDE Loss ###
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True, map=None):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y, map, training=False)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y, map, training=False)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, map=None, init_noise=None):
        if y is not None and batch_size != len(y):
            print(f'err ylen{len(y)}')
            raise ValueError("sample batch size different from length of given y")

        if init_noise is not None:
            print('initing noise.')
            x = init_noise
        else:
            x = torch.randn(batch_size, 3, *self.img_size, device=device)   
            init_noise = x
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema, map)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x[:, :3] #, init_noise
        # return x[:, :3] #.cpu().detach()
    
    @torch.no_grad()
    def edit_sample(self, batch_size, device, y=None, use_ema=True, map=None, mask=None, target=None):
        if y is not None and batch_size != len(y):
            print(f'err ylen{len(y)}')
            raise ValueError("sample batch size different from length of given y")


        x = torch.randn(batch_size, 3, *self.img_size, device=device)  

        raw = target

            
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema, map)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x[:, :3], raw[:, :3] 

    def finetune(self, batch_size, device, y=None, use_ema=True, map=None, init_noise=None):
        if y is not None and batch_size != len(y):
            print(f'err ylen{len(y)}')
            raise ValueError("sample batch size different from length of given y")

        if init_noise is not None:
            print('initing noise.')
            x = init_noise
        else:
            x = torch.randn(batch_size, 3, *self.img_size, device=device)
            init_noise = x
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema, map)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x[:, :3] #, init_noise

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, map=None, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            print(f'err ylen{len(y)}')
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, y, map, mask):
        # print(x.shape)
        noise = torch.randn(8, 3, 64, 64).cuda() #256
        # noise = torch.randn(1, 3, 128, 128).cuda() #512

        # ------ editing only --------
        # masked_x = x * (1-mask)/2
        # noise = noise * (1+mask)/2 + masked_x

        perturbed_x = self.perturb_x(x, t, noise)
        # ------ editing only --------
        # masked_x = x * (1+mask)/2
        # perturbed_x = perturbed_x * (1-mask)/2 + masked_x

        estimated_noise = self.model(perturbed_x, t, y, map)
        
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        if self.loss_type == "l1":
            loss_simple = F.l1_loss(estimated_noise, target)
        elif self.loss_type == "l2":
            loss_simple = F.mse_loss(estimated_noise, target)

        ## ---- SDE Loss ---
        
        loss_simple = F.mse_loss(estimated_noise, target, reduction='none').mean([1,2,3])

        logvar_t = self.logvar[t].cuda()
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        loss = loss.mean() * self.l_simple_weight

        loss_vlb = F.mse_loss(estimated_noise, target, reduction='none').mean(dim=(1,2,3))
        loss_vlb = (self.lvlb_weights[t] * loss).mean()

        loss += self.original_elbo_weight * loss_vlb

        return loss

    def forward(self, x, y=None, map=None, mask=None):
        b, c, h, w = x.shape
        device = x.device

        # if h != self.img_size[0]:
        #     raise ValueError("image height does not match diffusion parameters")
        # if w != self.img_size[0]:
        #     raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y, map, mask)
    
    def apply_model(self, x_noisy, t, cond, return_ids=False):
    
        # print(cond)
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            # x_recon = self.model(x_noisy, t, **cond)
            x_recon = self.model(x_noisy, t, map=cond['c_concat'][0])

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon



def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)