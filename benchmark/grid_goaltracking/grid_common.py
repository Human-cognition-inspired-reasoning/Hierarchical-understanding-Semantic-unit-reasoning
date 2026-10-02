import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

DPI = 100
EDGE, INK, BG = "#333333", "#222222", "#EEF0F3"


def canvas(width, height):
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=24, bold=True, ha="left", color=INK):
    ax.text(x, y, s, fontsize=size * 72 / DPI, fontweight="bold" if bold else "normal", ha=ha, va="center",
            color=color, zorder=5)


def cell(ax, x, y, w, fc):
    ax.add_patch(Rectangle((x, y), w, w, fc=fc, ec="#555555", lw=1, zorder=2))


def disc(ax, x, y, r, fc, ec=EDGE, lw=2, zorder=3):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=zorder))


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write(out, records):
    with open(out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_tile(name):
    a = np.asarray(Image.open(f"E:/fluent/png/{name}.png").convert("RGBA").resize((128, 128)))[..., 3] > 128
    ys, xs = np.nonzero(a)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return a.sum() / (h * w) >= 0.85 and 0.85 <= w / h <= 1.18 and h * w / 128 / 128 >= 0.55
