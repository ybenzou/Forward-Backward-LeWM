#!/usr/bin/env python3
"""Time-grid preview: one LeWM band (E/z/P) plus Forward. No cross-band losses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PINK = "#B64342"
PINK_FILL = "#F6DCDC"
BLUE = "#0F4D92"
BLUE_FILL = "#DCE6F2"
INK = "#1F1F1F"
MUTED = "#7A7A7A"
GUIDE = "#D0D0D0"
WHITE = "#FFFFFF"

OUT = Path(__file__).resolve().parent / "fig_temporal_leap_preview.png"
PAPER_PDF = Path(__file__).resolve().parents[1] / "figures" / "fig_has_training.pdf"


def box(ax, x, y, w, h, text, fc=WHITE, ec=INK, fs=11, lw=1.15, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=4,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight=weight, zorder=5)
    return dict(x=x, y=y, w=w, h=h)


def edge(node, side):
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    if side == "top":
        return (x, y + h / 2)
    if side == "bottom":
        return (x, y - h / 2)
    if side == "left":
        return (x - w / 2, y)
    if side == "right":
        return (x + w / 2, y)
    raise ValueError(side)


def arrow(ax, p0, p1, color=INK, lw=1.15, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
            shrinkA=0.4,
            shrinkB=0.4,
            zorder=2,
        )
    )


def polyarrow(ax, pts, color=INK, lw=1.15, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(
        xs,
        ys,
        color=color,
        lw=lw,
        ls=ls,
        solid_capstyle="round",
        solid_joinstyle="miter",
        zorder=2,
    )
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


def brace(ax, x_r, y_hi, y_lo, color, dx=0.36):
    """In-column comparison: prediction (top) vs target (bottom), right side."""
    x = x_r + dx
    ax.plot([x_r, x, x, x_r], [y_hi, y_hi, y_lo, y_lo], color=color, ls=(0, (2.2, 1.6)), lw=1.05, zorder=2)


def loss_to_z(ax, preds, z, color=BLUE, dx=0.32):
    """Dashed left-side link from Forward hats to the same-column z."""
    x = min(p["x"] - p["w"] / 2 for p in preds) - dx
    y_hi = max(p["y"] for p in preds)
    y_lo = z["y"]
    for p in preds:
        ax.plot([p["x"] - p["w"] / 2, x], [p["y"], p["y"]], color=color, ls=(0, (2.2, 1.6)), lw=1.05, zorder=2)
    ax.plot([x, x], [y_hi, y_lo], color=color, ls=(0, (2.2, 1.6)), lw=1.05, zorder=2)
    ax.plot([x, z["x"] - z["w"] / 2], [y_lo, y_lo], color=color, ls=(0, (2.2, 1.6)), lw=1.05, zorder=2)


def detach_mark(ax, x, y, color=BLUE):
    for d in (-0.05, 0.05):
        ax.plot([x - 0.07 + d, x + 0.07 + d], [y - 0.07, y + 0.07], color=color, lw=1.15, zorder=3)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.05), dpi=240)
    ax.set_xlim(0.15, 11.55)
    ax.set_ylim(0.00, 7.05)
    x_L = 10.62
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    T = [1.70, 4.40, 7.10, 9.80]
    Px = [(T[i] + T[i + 1]) / 2 for i in range(3)]

    y_x, y_E, y_z = 1.08, 1.74, 2.46
    y_rail = 2.86
    y_P, y_p = 3.58, 4.42
    y_step, y_roll = 5.50, 6.38
    P_w = 0.88
    tap_xs = [Px[i] - P_w / 2 - 0.10 for i in range(3)]

    ax.add_patch(Rectangle((0.45, 0.48), 10.55, 4.36, fc="#F8F1F1", ec="none", zorder=0))
    ax.add_patch(Rectangle((0.45, 4.96), 10.55, 1.78, fc="#F0F4F9", ec="none", zorder=0))
    ax.text(0.58, 2.70, "LeWM", color=PINK, fontsize=8, va="center")
    ax.text(0.58, 0.5 * (4.96 + 6.74), "Forward", color=BLUE, fontsize=8, va="center")

    ax.annotate(
        "",
        xy=(10.95, 0.26),
        xytext=(0.85, 0.26),
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0),
    )
    ax.text(11.08, 0.26, "time", color=MUTED, fontsize=9, va="center", style="italic")
    for t, x in enumerate(T):
        ax.plot([x, x], [0.48, 6.72], color=GUIDE, lw=0.7, ls=(0, (1.2, 2.2)), zorder=1)
        ax.plot([x, x], [0.18, 0.34], color=MUTED, lw=0.9, zorder=1)
        ax.text(x, 0.06, f"$t={t}$", ha="center", color=MUTED, fontsize=9)

    enc = box(ax, 5.75, y_E, 8.70, 0.48, r"$E$", fc=PINK_FILL, ec=PINK, fs=13, weight="bold")
    zs = []
    for t, x in enumerate(T):
        xb = box(ax, x, y_x, 0.68, 0.40, rf"$x_{t}$", fs=11)
        arrow(ax, edge(xb, "top"), (x, enc["y"] - enc["h"] / 2), color=PINK, lw=0.95)
        arrow(ax, (x, enc["y"] + enc["h"] / 2), (x, y_z - 0.22), color=PINK, lw=0.95)
        zs.append(box(ax, x, y_z, 0.68, 0.42, rf"$z_{t}$", fs=11))

    # Causal prefix bus inside LeWM: ends at P2's inlet, not under p2.
    for i in range(3):
        ax.plot([T[i], T[i]], [zs[i]["y"] + zs[i]["h"] / 2, y_rail], color=PINK, lw=1.05, zorder=2)
        ax.plot(T[i], y_rail, marker="o", ms=3.4, color=PINK, zorder=3)
    ax.plot([T[0], tap_xs[2]], [y_rail, y_rail], color=PINK, lw=1.05, zorder=2)

    a_labs = [r"$a_0$", r"$a_{0:1}$", r"$a_{0:2}$"]
    ps = []
    for i in range(3):
        P = box(ax, Px[i], y_P, P_w, 0.48, rf"$P_{i}$", fc=PINK_FILL, ec=PINK, fs=12, weight="bold")
        ax.text(Px[i], y_P - 0.38, a_labs[i], ha="center", color=PINK, fontsize=8)
        p = box(ax, T[i + 1], y_p, 0.68, 0.42, rf"$p_{i}$", fs=11)
        ps.append(p)
        polyarrow(ax, [(tap_xs[i], y_rail), (tap_xs[i], y_P), edge(P, "left")], color=PINK, lw=1.05)
        polyarrow(ax, [edge(P, "top"), (Px[i], y_p), edge(p, "left")], color=PINK, lw=1.05)
        brace(ax, T[i + 1] + 0.34, y_p, zs[i + 1]["y"], PINK)

    ax.text(x_L, y_p + 0.08, r"$\mathcal{L}_{\mathrm{LeWM}}$", color=PINK, fontsize=9, ha="left")
    ax.text(x_L, y_p - 0.16, r"Train $E,P$", color=PINK, fontsize=7, ha="left")

    # Forward: detached p lifts; hats compare to the same-column z below.
    F_s0 = box(ax, (T[1] + T[2]) / 2, y_step, 0.58, 0.40, r"$F$", fc=BLUE_FILL, ec=BLUE, fs=12, weight="bold")
    hat2 = box(ax, T[2], y_step, 0.78, 0.42, r"$\hat z_2$", fs=10)
    F_s1 = box(ax, (T[2] + T[3]) / 2, y_step, 0.58, 0.40, r"$F$", fc=BLUE_FILL, ec=BLUE, fs=12, weight="bold")
    hat3s = box(ax, T[3], y_step, 0.78, 0.42, r"$\hat z_3$", fs=10)

    arrow(ax, edge(ps[0], "top"), (T[1], y_step), color=BLUE, lw=1.1)
    arrow(ax, (T[1], y_step), edge(F_s0, "left"), color=BLUE, lw=1.1)
    arrow(ax, edge(F_s0, "right"), edge(hat2, "left"), color=BLUE, lw=1.1)
    y_sg = y_p + 0.50
    detach_mark(ax, T[1], y_sg)
    ax.text(T[1] - 0.36, y_sg, r"$\bar p_0$", color=BLUE, fontsize=8, ha="right", va="center")

    jy = y_step - 0.42
    arrow(ax, edge(ps[1], "top"), (T[2], jy), color=BLUE, lw=1.1)
    arrow(ax, (T[2], jy), (F_s1["x"], jy), color=BLUE, lw=1.1)
    arrow(ax, (F_s1["x"], jy), edge(F_s1, "bottom"), color=BLUE, lw=1.1)
    arrow(ax, edge(F_s1, "right"), edge(hat3s, "left"), color=BLUE, lw=1.1)
    detach_mark(ax, T[2], y_sg)
    ax.text(T[2] - 0.36, y_sg, r"$\bar p_1$", color=BLUE, fontsize=8, ha="right", va="center")

    ax.text(T[1] - 0.20, y_step, "step", color=BLUE, fontsize=8, ha="right", va="center")

    F_r0 = box(ax, (T[1] + T[2]) / 2, y_roll, 0.58, 0.40, r"$F$", fc=BLUE_FILL, ec=BLUE, fs=12, weight="bold")
    F_r1 = box(ax, (T[2] + T[3]) / 2, y_roll, 0.58, 0.40, r"$F$", fc=BLUE_FILL, ec=BLUE, fs=12, weight="bold")
    hat3r = box(ax, T[3], y_roll, 0.78, 0.42, r"$\hat z_3$", fs=10)

    arrow(ax, (T[1], y_step + 0.21), (T[1], y_roll), color=BLUE, lw=1.1)
    arrow(ax, (T[1], y_roll), edge(F_r0, "left"), color=BLUE, lw=1.1)
    arrow(ax, edge(F_r0, "right"), edge(F_r1, "left"), color=BLUE, lw=1.1)
    arrow(ax, edge(F_r1, "right"), edge(hat3r, "left"), color=BLUE, lw=1.1)

    loss_to_z(ax, [hat2], zs[2])
    loss_to_z(ax, [hat3s, hat3r], zs[3])
    ax.text(x_L, y_roll + 0.08, r"$\mathcal{L}_{\mathrm{roll}}$", color=BLUE, fontsize=9, ha="left")
    ax.text(x_L, y_roll - 0.16, r"Train $F$", color=BLUE, fontsize=7, ha="left")
    ax.text(T[1] - 0.20, y_roll, "roll", color=BLUE, fontsize=8, ha="right", va="center")
    ax.text(x_L, y_step + 0.08, r"$\mathcal{L}_{\mathrm{step}}$", color=BLUE, fontsize=9, ha="left")
    ax.text(x_L, y_step - 0.16, r"Train $F$", color=BLUE, fontsize=7, ha="left")

    fig.savefig(OUT, dpi=240, bbox_inches="tight", pad_inches=0.07, facecolor="white")
    fig.savefig(PAPER_PDF, bbox_inches="tight", pad_inches=0.07, facecolor="white")
    plt.close(fig)
    print("wrote", OUT)
    print("wrote", PAPER_PDF)


if __name__ == "__main__":
    main()
