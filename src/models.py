from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _unwrap_checkpoint_state(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ["model_state", "state_dict", "model", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format")
    return ckpt


def _strip_prefixes(state: dict, prefixes: tuple[str, ...]) -> dict:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _infer_prithvi_vit_name(checkpoint_path: str, fallback: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _unwrap_checkpoint_state(ckpt)
    state = _strip_prefixes(state, ("encoder.", "backbone.", "module."))
    return infer_vit_name_from_state_dict(state, fallback)


def infer_vit_name_from_state_dict(state: dict, fallback: str) -> str:
    patch_weight = state.get("patch_embed.proj.weight")
    if patch_weight is None:
        patch_weight = state.get("encoder.patch_embed.proj.weight")
    if patch_weight is None:
        return fallback

    embed_dim = int(patch_weight.shape[0])
    if embed_dim == 192:
        return "vit_tiny_patch16_224"
    if embed_dim == 768:
        return "vit_base_patch16_224"
    return fallback


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x //
                   2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetRegressor(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.drop = nn.Dropout2d(dropout)
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bottleneck = self.bottleneck(x4)
        xb = self.drop(bottleneck)
        return x1, x2, x3, x4, xb, bottleneck

    def decode(self, feats):
        x1, x2, x3, x4, xb, _ = feats
        x = self.up1(xb, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def forward(self, x, return_features: bool = False):
        feats = self.encode(x)
        y = self.decode(feats)
        y = torch.sigmoid(y)
        if return_features:
            return y, feats[-1]
        return y


class ViTDenseRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 128,
        vit_name: str = "vit_base_patch16_224",
        pretrained_checkpoint: Optional[str] = None,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.vit_name = vit_name
        self.encoder = timm.create_model(
            vit_name,
            pretrained=False,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = getattr(self.encoder, "num_features", 768)
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.grid_size = img_size // self.patch_size
        self.norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim //
                      2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim //
                      4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 4, 1, kernel_size=1),
        )
        if pretrained_checkpoint:
            self._load_pretrained(pretrained_checkpoint)

    def _adapt_patch_embed(self, weight: torch.Tensor, in_channels: int) -> torch.Tensor:
        if weight.shape[1] == in_channels:
            return weight
        if weight.shape[1] > in_channels:
            return weight[:, :in_channels, :, :]
        repeat = (in_channels + weight.shape[1] - 1) // weight.shape[1]
        weight = weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
        weight = weight * (weight.shape[1] / in_channels)
        return weight

    def _load_pretrained(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt = _unwrap_checkpoint_state(ckpt)

        state = {}
        for k, v in ckpt.items():
            nk = k
            for prefix in ["encoder.", "backbone.", "module."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            state[nk] = v

        patch_key = "patch_embed.proj.weight"
        if patch_key in state:
            state[patch_key] = self._adapt_patch_embed(
                state[patch_key], self.encoder.patch_embed.proj.weight.shape[1])

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"Loaded pretrained encoder from {checkpoint_path}")
        print(
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def encode(self, x):
        B = x.shape[0]
        tokens = self.encoder.forward_features(x)
        if tokens.ndim == 3:
            if tokens.shape[1] == self.grid_size * self.grid_size + 1:
                tokens = tokens[:, 1:, :]
            tokens = self.norm(tokens)
            feat = tokens.transpose(1, 2).reshape(
                B, self.embed_dim, self.grid_size, self.grid_size)
        else:
            raise ValueError(f"Unexpected token tensor shape: {tokens.shape}")
        return feat

    def forward(self, x, return_features: bool = False):
        feat = self.encode(x)
        y = self.head(self.drop(feat))
        y = F.interpolate(y, size=x.shape[-2:],
                          mode="bilinear", align_corners=False)
        y = torch.sigmoid(y)
        if return_features:
            return y, feat
        return y


class ViTDensePrithvi(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 128,
        vit_name: str = "vit_base_patch16_224",
        pretrained_checkpoint: str = None,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.vit_name = vit_name
        self.encoder = timm.create_model(
            vit_name,
            pretrained=False,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

        self.embed_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.grid_size = img_size // self.patch_size

        self.norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 4, 1, 1),
        )
        if pretrained_checkpoint:
            self._load_prithvi(pretrained_checkpoint)

    def _load_prithvi(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = _unwrap_checkpoint_state(ckpt)

        state = {}

        for k, v in ckpt.items():
            k = k.replace("encoder.", "").replace(
                "backbone.", "").replace("module.", "")

            if (
                "pos_embed" in k
                or "cls_token" in k
                or "temporal_embed" in k
                or "location_embed" in k
                or k.startswith("decoder.")
            ):
                continue

            state[k] = v

        patch_key = "patch_embed.proj.weight"
        if patch_key in state:
            if state[patch_key].ndim == 5:
                if state[patch_key].shape[2] != 1:
                    raise ValueError(
                        f"Unsupported temporal patch depth in {ckpt_path}: {tuple(state[patch_key].shape)}"
                    )
                state[patch_key] = state[patch_key][:, :, 0, :, :]
            state[patch_key] = self._adapt_patch_embed(
                state[patch_key],
                self.encoder.patch_embed.proj.weight.shape[1],
            )

        model_state = self.encoder.state_dict()
        filtered_state = {}
        skipped_shape = []
        for key, value in state.items():
            if key not in model_state:
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                skipped_shape.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value

        missing, unexpected = self.encoder.load_state_dict(
            filtered_state, strict=False)

        print("\nPrithvi checkpoint loaded")
        print(f"Loaded keys: {len(filtered_state)}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}\n")
        if skipped_shape:
            print("Skipped incompatible keys:")
            for key, found_shape, expected_shape in skipped_shape[:10]:
                print(
                    f"  {key}: checkpoint {found_shape} vs model {expected_shape}")
            if len(skipped_shape) > 10:
                print(f"  ... and {len(skipped_shape) - 10} more")

    def _adapt_patch_embed(self, weight, in_channels):
        if weight.shape[1] == in_channels:
            return weight

        if weight.shape[1] > in_channels:
            return weight[:, :in_channels, :, :]

        repeat = (in_channels + weight.shape[1] - 1) // weight.shape[1]
        weight = weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
        weight = weight * (weight.shape[1] / in_channels)
        return weight

    def encode(self, x):
        B = x.shape[0]

        tokens = self.encoder.forward_features(x)

        if tokens.ndim == 3:
            if tokens.shape[1] == self.grid_size * self.grid_size + 1:
                tokens = tokens[:, 1:, :]

            tokens = self.norm(tokens)

            feat = tokens.transpose(1, 2).reshape(
                B,
                self.embed_dim,
                self.grid_size,
                self.grid_size
            )
        else:
            raise ValueError(f"Unexpected shape: {tokens.shape}")

        return feat

    def forward(self, x, return_features: bool = False):
        feat = self.encode(x)

        y = self.head(self.drop(feat))

        y = F.interpolate(
            y,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        y = torch.sigmoid(y)

        if return_features:
            return y, feat

        return y


def build_model(
    model_name: str,
    in_channels: int,
    patch_size: int,
    pretrained_checkpoint: Optional[str] = None,
    vit_name: str = "vit_base_patch16_224",
):
    model_name = model_name.lower()
    if model_name in {"prithvi", "geo_vit"} and pretrained_checkpoint:
        inferred_vit_name = _infer_prithvi_vit_name(
            pretrained_checkpoint, vit_name)
        if inferred_vit_name != vit_name:
            print(
                f"Switching ViT backbone from {vit_name} to {inferred_vit_name} "
                f"to match checkpoint {Path(pretrained_checkpoint).name}"
            )
            vit_name = inferred_vit_name
    if model_name == "unet":
        return UNetRegressor(in_channels=in_channels, base=32, dropout=0.1)
    if model_name in {"vit"}:
        return ViTDenseRegressor(
            in_channels=in_channels,
            img_size=patch_size,
            vit_name=vit_name,
            pretrained_checkpoint=pretrained_checkpoint,
        )
    if model_name in {"prithvi", "geo_vit"}:
        return ViTDensePrithvi(
            in_channels=in_channels,
            img_size=patch_size,
            vit_name=vit_name,
            pretrained_checkpoint=pretrained_checkpoint,
        )

    raise ValueError(f"Unknown model: {model_name}")
