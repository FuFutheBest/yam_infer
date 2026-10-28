"""ACT: Action Chunking with Transformers (Zhao et al. 2023), multi-task variant.

Inputs  : 2 wrist RGB images, follower qpos (14), task id
Output  : chunk of K future leader joint targets (K, 14), normalised space

           ┌───────── CVAE encoder (training only) ─────────┐
 actions ─►│ [CLS, qpos, a_0..a_K-1] → Transformer → (mu, σ) │──► z (32)
           └────────────────────────────────────────────────┘     (z = 0 at inference)
 images ─► ResNet18 ─► 8×10 feature tokens per camera ┐
 qpos, z, task ───────────────────────────────────────┴► Transformer encoder ─► memory
 K learned queries ─► Transformer decoder (cross-attends memory) ─► Linear ─► K×14 actions
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ACTConfig:
    state_dim: int = 14
    action_dim: int = 14
    n_cameras: int = 2
    n_tasks: int = 2
    chunk_size: int = 50
    d_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    backbone: str = "resnet18"
    pretrained_backbone: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoid_1d(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sinusoid_2d(h: int, w: int, d: int) -> torch.Tensor:
    """(h*w, d): first half encodes y, second half encodes x."""
    half = d // 2
    pe_y = sinusoid_1d(h, half)[:, None, :].expand(h, w, half)
    pe_x = sinusoid_1d(w, half)[None, :, :].expand(h, w, half)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, d)


def _resnet_backbone(name: str, pretrained: bool) -> nn.Module:
    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d

    weights = "IMAGENET1K_V1" if pretrained else None
    net = getattr(torchvision.models, name)(
        weights=weights, norm_layer=FrozenBatchNorm2d
    )
    return nn.Sequential(
        *list(net.children())[:-2]
    )  # drop avgpool + fc -> (B, 512, H/32, W/32)


def _enc_layer(cfg: ACTConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        cfg.d_model,
        cfg.n_heads,
        cfg.dim_feedforward,
        cfg.dropout,
        batch_first=True,
        norm_first=True,
    )


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.backbone = _resnet_backbone(cfg.backbone, cfg.pretrained_backbone)
        self.input_proj = nn.Conv2d(512, d, kernel_size=1)
        self.cam_embed = nn.Parameter(torch.zeros(cfg.n_cameras, d))
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.latent_proj = nn.Linear(cfg.latent_dim, d)
        self.task_embed = nn.Embedding(cfg.n_tasks, d)
        self.extra_pos = nn.Parameter(torch.zeros(3, d))  # latent, state, task tokens

        self.encoder = nn.TransformerEncoder(
            _enc_layer(cfg), cfg.n_encoder_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d,
                cfg.n_heads,
                cfg.dim_feedforward,
                cfg.dropout,
                batch_first=True,
                norm_first=True,
            ),
            cfg.n_decoder_layers,
        )
        self.query_embed = nn.Parameter(torch.randn(cfg.chunk_size, d) * 0.02)
        self.memory_norm = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, cfg.action_dim)

        # CVAE encoder: compresses the demonstrated chunk into a "style" latent z.
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.vae_state_proj = nn.Linear(cfg.state_dim, d)
        self.vae_action_proj = nn.Linear(cfg.action_dim, d)
        self.vae_encoder = nn.TransformerEncoder(
            _enc_layer(cfg), cfg.n_vae_layers, enable_nested_tensor=False
        )
        self.vae_out = nn.Linear(d, 2 * cfg.latent_dim)
        self.vae_pos: torch.Tensor
        self.img_mean: torch.Tensor
        self.img_std: torch.Tensor
        self.register_buffer(
            "vae_pos", sinusoid_1d(cfg.chunk_size + 2, d), persistent=False
        )
        self.register_buffer(
            "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # ------------------------------------------------------------------ helpers
    def backbone_parameters(self):
        return self.backbone.parameters()

    def other_parameters(self):
        ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: float (B, n_cam, 3, H, W) in [0,1]  ->  tokens (B, n_cam*h*w, d)."""
        b, n, c, h, w = images.shape
        x = (images.reshape(b * n, c, h, w) - self.img_mean) / self.img_std
        feat = self.input_proj(self.backbone(x))  # (B*n, d, fh, fw)
        fh, fw = feat.shape[-2:]
        pos = sinusoid_2d(fh, fw, self.cfg.d_model).to(feat.device, feat.dtype)
        tokens = feat.flatten(2).transpose(1, 2) + pos  # (B*n, fh*fw, d)
        tokens = tokens.reshape(b, n, fh * fw, -1) + self.cam_embed[
            None, :, None, :
        ].to(tokens.dtype)
        return tokens.reshape(b, n * fh * fw, -1)

    def encode_latent(self, qpos: torch.Tensor, actions: torch.Tensor):
        b = qpos.shape[0]
        tokens = torch.cat(
            [
                self.vae_cls.expand(b, -1, -1),
                self.vae_state_proj(qpos)[:, None],
                self.vae_action_proj(actions),
            ],
            dim=1,
        )
        tokens = tokens + self.vae_pos[None, : tokens.shape[1]]
        out = self.vae_encoder(tokens)[:, 0]
        mu, logvar = self.vae_out(out).chunk(2, dim=-1)
        return mu, logvar

    # ------------------------------------------------------------------ forward
    def forward(self, images, qpos, task, actions=None):
        """Returns (pred_actions (B,K,A), mu, logvar). mu/logvar are None at inference."""
        b = qpos.shape[0]
        if actions is not None:
            mu, logvar = self.encode_latent(qpos, actions)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(
                b, self.cfg.latent_dim, device=qpos.device, dtype=qpos.dtype
            )

        extra = (
            torch.stack(
                [self.latent_proj(z), self.state_proj(qpos), self.task_embed(task)],
                dim=1,
            )
            + self.extra_pos[None]
        )
        memory = self.encoder(torch.cat([extra, self.encode_images(images)], dim=1))
        memory = self.memory_norm(memory)
        queries = self.query_embed[None].expand(b, -1, -1)
        out = self.out_norm(self.decoder(queries, memory))
        return self.action_head(out), mu, logvar


def act_loss(pred, target, is_pad, mu, logvar, kl_weight: float):
    l1 = (F.l1_loss(pred, target, reduction="none").mean(-1) * (~is_pad)).sum() / (
        ~is_pad
    ).sum().clamp(min=1)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl
