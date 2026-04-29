from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "figure_unet_architecture.png"


def add_box(ax, x, y, w, h, text, fc, ec="#2f3b52", fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color="#34495e", lw=1.8, style="->"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, linewidth=lw, color=color),
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    enc_fc = "#dceeff"
    bottleneck_fc = "#fde7d9"
    dec_fc = "#dff3e4"
    out_fc = "#f8d7da"
    skip_color = "#7d3c98"

    y_top = 5.2
    y_bot = 1.8
    w = 1.7
    h = 1.0

    encoder = [
        (0.7, y_top, "Input\n5 x 128 x 128"),
        (2.7, y_top, "DoubleConv\n32 x 128 x 128"),
        (4.8, y_top, "Down\n64 x 64 x 64"),
        (6.9, y_top, "Down\n128 x 32 x 32"),
        (9.0, y_top, "Down\n256 x 16 x 16"),
    ]
    for x, y, txt in encoder:
        add_box(ax, x, y, w, h, txt, enc_fc)

    add_box(ax, 11.1, y_top, 2.0, h, "Bottleneck\nDoubleConv\n512 x 16 x 16\nDropout 0.1", bottleneck_fc)

    decoder = [
        (9.3, y_bot, "Up1\n256 x 16 x 16"),
        (7.2, y_bot, "Up2\n128 x 32 x 32"),
        (5.1, y_bot, "Up3\n64 x 64 x 64"),
        (3.0, y_bot, "Up4\n32 x 128 x 128"),
        (0.8, y_bot, "1x1 Conv\nSigmoid\n1 x 128 x 128"),
    ]
    for x, y, txt in decoder[:-1]:
        add_box(ax, x, y, w, h, txt, dec_fc)
    add_box(ax, decoder[-1][0], decoder[-1][1], 1.9, h, decoder[-1][2], out_fc)

    # Main forward arrows
    for i in range(len(encoder) - 1):
        arrow(ax, encoder[i][0] + w, y_top + h / 2, encoder[i + 1][0], y_top + h / 2)
    arrow(ax, encoder[-1][0] + w, y_top + h / 2, 11.1, y_top + h / 2)
    arrow(ax, 11.1, y_top, 10.1, y_bot + h, color="#34495e")
    for i in range(len(decoder) - 1):
        x, y, _ = decoder[i]
        nx, ny, _ = decoder[i + 1]
        arrow(ax, x, y + h / 2, nx + (1.9 if i + 1 == len(decoder) - 1 else w), ny + h / 2)

    # Skip connections
    skip_pairs = [
        ((2.7 + w / 2, y_top), (3.0 + w / 2, y_bot + h)),
        ((4.8 + w / 2, y_top), (5.1 + w / 2, y_bot + h)),
        ((6.9 + w / 2, y_top), (7.2 + w / 2, y_bot + h)),
        ((9.0 + w / 2, y_top), (9.3 + w / 2, y_bot + h)),
    ]
    for (x1, y1), (x2, y2) in skip_pairs:
        arrow(ax, x1, y1, x2, y2, color=skip_color, lw=1.8)

    ax.text(7.9, 7.35, "U-Net Architecture Used in This Project", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(7.9, 6.85, "4-level encoder-decoder regressor with skip connections for 128 x 128 raster patches", ha="center", va="center", fontsize=11)

    ax.text(13.7, 5.95, "Encoder", fontsize=11, color="#1b4f72", fontweight="bold")
    ax.text(13.7, 5.55, "Conv3x3 + BN + ReLU\nrepeated twice per block", fontsize=9, va="top")
    ax.text(13.7, 4.55, "Down blocks:\nMaxPool2d(2) + DoubleConv", fontsize=9, va="top")
    ax.text(13.7, 3.55, "Up blocks:\nConvTranspose2d + skip concat\n+ DoubleConv", fontsize=9, va="top")
    ax.text(13.7, 2.35, "Output:\ncontinuous canopy fraction\nscaled later to percent", fontsize=9, va="top")

    ax.text(11.9, 1.1, "Purple arrows = skip connections from encoder to decoder", fontsize=9, color=skip_color, ha="center")

    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
