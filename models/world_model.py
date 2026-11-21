"""Step 5: the world model. A diffusion denoiser that draws frame t+1 from frames t-3..t and actions t-3..t.

Adapted from DIAMOND (github.com/eloialonso/diamond, src/models/diffusion/{inner_model,denoiser,diffusion_sampler}.py,
MIT, see models/DIAMOND_LICENSE). Same EDM maths and U-Net; two changes (DECISIONS 2026-11-21):
  1. actions: Pusher has 7 continuous torques, so the last 4 actions (28 numbers) go through a small MLP
     instead of DIAMOND's nn.Embedding lookup table for discrete Atari buttons;
  2. context noise augmentation (GameNGen): in training the 4 past frames get random noise of strength
     aug in [0, aug_max], and aug is one more hint, so the model learns not to trust its context blindly.
Frames come in as uint8 (B, 4, 3, 64, 64) / (B, 3, 64, 64) and are scaled to [-1, 1] here, and only here.

Self-test:  python -m models.world_model --config configs/world_model_v1.yaml
"""
import argparse
import time

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from envs.constants import ACTION_DIM, ACTION_LIMIT, CONTEXT_FRAMES, FRAME_SIZE
from models.diamond_blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


def to_model(frames: Tensor) -> Tensor:
    """uint8 [0, 255] -> float [-1, 1]."""
    return frames.float() / 127.5 - 1


def to_uint8(x: Tensor) -> Tensor:
    """float [-1, 1] -> uint8 [0, 255] (rounded, clipped)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def add_dims(x: Tensor, n: int) -> Tensor:
    """(B,) -> (B, 1, 1, 1) so a per-example number can multiply a batch of images."""
    return x.reshape(x.shape + (1,) * (n - x.ndim))


class InnerModel(nn.Module):
    """The U-Net with its hints: noisy frame + 4 context frames in, a 'correction' image out."""

    def __init__(self, cond_channels: int, depths: list[int], channels: list[int], attn_depths: list[int]) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cond_channels)        # noise level sigma -> vector
        self.aug_emb = FourierFeatures(cond_channels)          # NEW: context-noise level -> vector
        self.act_emb = nn.Sequential(                          # NEW: 4 x 7 torques -> vector (DIAMOND: nn.Embedding)
            nn.Flatten(), nn.Linear(CONTEXT_FRAMES * ACTION_DIM, cond_channels), nn.SiLU(),
            nn.Linear(cond_channels, cond_channels))
        self.cond_proj = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(),
                                       nn.Linear(cond_channels, cond_channels))
        self.conv_in = Conv3x3((CONTEXT_FRAMES + 1) * 3, channels[0])     # 15 channels in
        self.unet = UNet(cond_channels, depths, channels, attn_depths)
        self.norm_out = GroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], 3)
        nn.init.zeros_(self.conv_out.weight)                   # start by predicting "no correction"

    def forward(self, noisy: Tensor, c_noise: Tensor, ctx: Tensor, act: Tensor, c_aug: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise) + self.aug_emb(c_aug) + self.act_emb(act / ACTION_LIMIT))
        x = self.conv_in(torch.cat((ctx, noisy), dim=1))
        x, _, _ = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))


class WorldModel(nn.Module):
    """EDM denoiser around InnerModel: loss() for training, sample() to draw the next frame."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m, d = cfg["model"], cfg["diffusion"]
        self.inner = InnerModel(m["cond_channels"], m["depths"], m["channels"], m["attn_depths"])
        self.sigma_data, self.sigma_offset = d["sigma_data"], d["sigma_offset_noise"]
        self.sigma_dist = d["sigma_distribution"]
        self.aug_max = cfg["training"]["noise_augmentation_max"]

    def conditioners(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """EDM preconditioning: how to scale input and output at noise level sigma (B,)."""
        sigma = (sigma**2 + self.sigma_offset**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()                  # input -> roughly unit size
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)      # how much of the noisy frame to keep
        c_out = sigma * c_skip.sqrt()                                      # how big the network's correction is
        c_noise = sigma.log() / 4
        return add_dims(c_in, 4), add_dims(c_skip, 4), add_dims(c_out, 4), c_noise

    def inner_output(self, noisy: Tensor, sigma: Tensor, ctx: Tensor, act: Tensor, aug: Tensor) -> Tensor:
        """Raw network output for noisy frame (B, 3, 64, 64) at sigma (B,); ctx (B, 12, 64, 64) already in [-1, 1]."""
        c_in, _, _, c_noise = self.conditioners(sigma)
        return self.inner(noisy * c_in, c_noise, ctx / self.sigma_data, act, aug)

    def loss(self, ctx_u8: Tensor, act: Tensor, target_u8: Tensor) -> Tensor:
        """One training step's loss: bury the true next frame in noise, grade the network's clean-up."""
        b = len(target_u8)
        ctx, target = to_model(ctx_u8).flatten(1, 2), to_model(target_u8)  # (B, 12, 64, 64), (B, 3, 64, 64)
        aug = torch.rand(b, device=ctx.device) * self.aug_max               # context noise strength per example
        ctx = ctx + add_dims(aug, 4) * torch.randn_like(ctx)
        s = self.sigma_dist                                                 # log-normal noise levels
        sigma = (torch.randn(b, device=ctx.device) * s["scale"] + s["loc"]).exp().clip(s["sigma_min"], s["sigma_max"])
        offset = self.sigma_offset * torch.randn(b, 3, 1, 1, device=ctx.device)   # whole-channel shift (DIAMOND)
        noisy = target + offset + add_dims(sigma, 4) * torch.randn_like(target)
        _, c_skip, c_out, _ = self.conditioners(sigma)
        goal = (target - c_skip * noisy) / c_out                            # the correction that gives the clean frame
        return F.mse_loss(self.inner_output(noisy, sigma, ctx, act, aug), goal)

    @torch.no_grad()
    def denoise(self, noisy: Tensor, sigma: Tensor, ctx: Tensor, act: Tensor, aug: Tensor) -> Tensor:
        """The model's guess of the clean frame, rounded to 256 levels like a real uint8 frame."""
        _, c_skip, c_out, _ = self.conditioners(sigma)
        x = c_skip * noisy + c_out * self.inner_output(noisy, sigma, ctx, act, aug)
        return to_model(to_uint8(x))

    @torch.no_grad()
    def sample(self, ctx_u8: Tensor, act: Tensor, steps: int = 3, aug_level: float = 0.0,
               sigma_min: float = 2e-3, sigma_max: float = 5.0, rho: float = 7.0) -> Tensor:
        """Draw frame t+1 (B, 3, 64, 64) uint8 from pure noise with `steps` Euler steps (DIAMOND's sampler)."""
        b, dev = len(ctx_u8), ctx_u8.device
        ctx = to_model(ctx_u8).flatten(1, 2)
        aug = torch.full((b,), aug_level, device=dev)
        ramp = torch.linspace(0, 1, steps, device=dev)                     # big steps at high noise, small at low
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        x = torch.randn(b, 3, FRAME_SIZE, FRAME_SIZE, device=dev)   # unit noise, exactly as DIAMOND (not x sigma_max)
        for sigma, nxt in zip(sigmas[:-1], sigmas[1:]):
            denoised = self.denoise(x, sigma.expand(b), ctx, act, aug)
            x = x + (x - denoised) / sigma * (nxt - sigma)               # Euler: move toward the clean guess
        return to_uint8(x)


def self_test(cfg: dict) -> None:
    """Shapes, parameter count, a 40-step overfit on one batch (loss must fall), and one sampled frame."""
    torch.manual_seed(0)
    model = WorldModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    b = 8
    ctx = torch.randint(0, 256, (b, CONTEXT_FRAMES, 3, FRAME_SIZE, FRAME_SIZE), dtype=torch.uint8)
    act = (torch.rand(b, CONTEXT_FRAMES, ACTION_DIM) * 2 - 1) * ACTION_LIMIT
    target = ctx[:, -1].clone()                                            # easy task: "next frame = newest frame"
    print(f"world model: {n_params:,} parameters (U-Net {sum(p.numel() for p in model.inner.unet.parameters()):,})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"] * 10)
    losses, t0 = [], time.time()
    for _ in range(40):
        loss = model.loss(ctx, act, target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"overfit one batch: loss {first:.3f} -> {last:.3f} in 40 steps ({(time.time() - t0) / 40:.2f} s/step on CPU)")
    assert last < first, "loss did not fall on a single repeated batch"

    model.eval()
    out = model.sample(ctx, act, steps=cfg["sampling"]["denoising_steps"])
    assert out.shape == (b, 3, FRAME_SIZE, FRAME_SIZE) and out.dtype == torch.uint8
    err = (out.float() - target.float()).abs().mean().item()
    print(f"sample OK: {tuple(out.shape)} {out.dtype} | mean |sample - target| {err:.1f} of 255 (random noise ~85)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    self_test(yaml.safe_load(open(parser.parse_args().config)))
