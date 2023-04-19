# Copyright 2023 solo-learn development team.
import copy
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from solo.losses.simclr import simclr_loss_func, my_loss_func
from solo.methods.base import BaseMethod


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    result = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return result

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps = 1000,
        beta_schedule = 'cosine',
    ):
        super().__init__()

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas.to('cuda'))
        register_buffer('alphas_cumprod', alphas_cumprod.to('cuda'))
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to('cuda'))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to('cuda'))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to('cuda'))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to('cuda'))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance.to('cuda'))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)).to('cuda'))
        register_buffer('posterior_mean_coef1', (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to('cuda'))
        register_buffer('posterior_mean_coef2', ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).to('cuda'))

        # calculate p2 reweighting

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.diffusion_projector = GaussianDiffusion(timesteps=10)
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]
        index_all = []
        data_modified = []
        t_all = []

        # diff_num = int(len(indexes)/50)
        # noise_num = 2

        diff_num = 10
        noise_num = 0
        t_ = 9

        # diff_num = 0
        # noise_num = 7

        for data in batch[1]:
            noise = torch.randn_like(data)
            # noise_ = torch.rand_like(data)

            # t = torch.randint(low=3, high=4, size=(data.shape[0],), dtype=torch.int64)
            t_original = torch.zeros(data.shape[0], dtype = torch.int64)
            t_original = t_original.to('cuda')
            t = torch.randint(low=0, high=2, size=(data.shape[0],), dtype=torch.int64)
            t = torch.where(t == 0, torch.tensor(t_), torch.tensor(t_))
            # t = torch.ones(data.shape[0], dtype=torch.int64)

            t = t.to('cuda')
            diff_data = self.diffusion_projector.q_sample(x_start=data, t=t, noise=noise)

            index_diff = copy.deepcopy(indexes)
            index_diff = -index_diff

            index_noise = torch.ones_like(indexes)
            index_noise = -99999999 * index_noise

            t_noise = torch.ones_like(index_noise)
            t_noise = t_noise.to('cuda')
            t_noise *= 999

            # index_noise_ = torch.ones_like(indexes)
            # index_noise_ = -88888888 * index_noise_

            index_all.append(indexes)
            index_all.append(index_diff[:diff_num].to('cuda'))
            index_all.append(index_noise[:noise_num])

            t_all.append(torch.cat([t_original, t[:diff_num], t_noise[:noise_num]]))
            data_modified.append(torch.cat([data, diff_data[:diff_num], noise[:noise_num]]))

        batch[1] = data_modified
        batch[2] = torch.cat([batch[2],batch[2][:diff_num], batch[2][:noise_num]])
        # batch[2] = torch.cat([batch[2], batch[2], batch[2][:2]])
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        # n_augs = self.num_large_crops + self.num_small_crops
        # indexes = indexes.repeat(n_augs)


        indexes = torch.cat(index_all)

        # z_list = out['z']
        # z_all = []
        # for one in z_list:
        #     # original = one[:batch_size]
        #     noise = one[-2:]
        #     # z_all.append(original)
        #     z_all.append(noise)

        nce_loss = simclr_loss_func(
            # torch.cat(z_all),
            z,
            indexes=indexes,
            t_all = t_all,
            temperature=self.temperature,
        )

        # my_loss1 = my_loss_func(out['z'])
        # my_loss2 = torch.norm(torch.cat(z_all))
        # var = torch.var(torch.cat(z_all))

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        # self.log('my_loss1', my_loss1)
        # self.log('my_loss2', my_loss2)
        # self.log('var_noise', var)
        return nce_loss + class_loss

    def training_step_old(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
