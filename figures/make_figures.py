#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""교재 그림 생성 — 데이터가 들어가는 그림은 전부 여기서 만든다.

    python figures/make_figures.py

원칙
  · 인용 그림을 쓰지 않는다. 전부 새로 그린다 (저작권 문제도 함께 해소).
  · SVG로 낸다. 인쇄본은 벡터 그대로, 온라인판도 같은 파일을 쓴다.
  · 색만으로 구분하지 않는다. 마커 모양도 함께 다르게 한다 (색각 이상 대응).
  · 그림에 쓰인 데이터는 dlbook.data에서 온다. 본문의 숫자와 어긋날 수 없다.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib  # noqa: F401  한글 폰트
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from dlbook import data, metrics  # noqa: E402

OUT = ROOT
OUT.mkdir(exist_ok=True)

# ── 팔레트 ────────────────────────────────────────────────────────────────
RIPE, UNRIPE = "#C1443C", "#2E6F95"      # 익은 것 / 안 익은 것
LINE, LINE2 = "#26282B", "#9AA3AB"        # 기준선 (현재 / 이전)
FILL_R, FILL_U = "#F4DAD7", "#DCE8F0"     # 영역 채우기
ACC = "#B8860B"                            # 강조

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": "#4A4F55",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#26282B",
    "text.color": "#26282B",
    "xtick.color": "#4A4F55",
    "ytick.color": "#4A4F55",
    "grid.color": "#DFE3E7",
    "svg.fonttype": "none",   # 텍스트를 글자로 남긴다 (편집·검색 가능)
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name: str):
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {path.name}")
    return path


def scatter(ax, x, y, s=30, legend=True):
    """익은 것은 붉은 원, 안 익은 것은 푸른 삼각형. 색과 모양을 함께 쓴다."""
    m0, m1 = y == 0, y == 1
    ax.scatter(x[m0, 0], x[m0, 1], s=s, c=UNRIPE, marker="^",
               edgecolors="white", linewidths=0.5, label="안 익음", zorder=3)
    ax.scatter(x[m1, 0], x[m1, 1], s=s, c=RIPE, marker="o",
               edgecolors="white", linewidths=0.5, label="익음", zorder=3)
    if legend:
        ax.legend(loc="lower left", framealpha=0.9, fontsize=9)


def draw_line(ax, w1, w2, b, xlim, color=LINE, ls="-", lw=2.0, label=None, z=2):
    """w1·x1 + w2·x2 + b = 0 을 그린다."""
    xs = np.linspace(*xlim, 100)
    if abs(w2) < 1e-9:
        ax.axvline(-b / w1, color=color, ls=ls, lw=lw, label=label, zorder=z)
    else:
        ax.plot(xs, -(w1 * xs + b) / w2, color=color, ls=ls, lw=lw,
                label=label, zorder=z)


def shade(ax, w1, w2, b, xlim, ylim, res=300, alpha=1.0):
    gx, gy = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, res))
    zz = w1 * gx + w2 * gy + b > 0
    ax.contourf(gx, gy, zz.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=[FILL_U, FILL_R], alpha=alpha, zorder=0)


# ══ 그림 2-1 ══ 1차원 수직선과 기준점 ═══════════════════════════════════════
def fig_2_1():
    x, y = data.tangerines(40, seed=3)
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.axhline(0, color="#4A4F55", lw=1.2, zorder=1)
    jitter = np.random.default_rng(0).normal(0, 0.035, len(x))
    for cls, c, mk, lb in ((0, UNRIPE, "^", "안 익음"), (1, RIPE, "o", "익음")):
        m = y == cls
        ax.scatter(x[m, 0], jitter[m], c=c, marker=mk, s=44,
                   edgecolors="white", linewidths=0.5, label=lb, zorder=3)
    ax.axvline(7.5, color=LINE, lw=2.2, ymin=0.18, ymax=0.82, zorder=2)
    ax.annotate("기준 7.5cm", xy=(7.5, 0.20), xytext=(7.5, 0.34),
                ha="center", fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=LINE, lw=1))
    ax.text(4.2, -0.30, "← 작다", fontsize=10, color="#4A4F55")
    ax.text(10.3, -0.30, "크다 →", fontsize=10, color="#4A4F55", ha="right")
    ax.set_xlim(3.6, 11.2); ax.set_ylim(-0.42, 0.46)
    ax.set_yticks([]); ax.set_xlabel("크기 (cm)")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    return save(fig, "fig_2_01_1차원_기준점")


# ══ 그림 2-2 ══ 기준이 옮겨 가는 4컷 ════════════════════════════════════════
def fig_2_2():
    x, y = data.tangerines(28, seed=5)
    xs = x[:, 0]
    thresholds = [6.0, 6.7, 7.2, 7.5]
    caps = ["① 아무 데나 잡는다", "② 틀린 것이 있다 → 민다",
            "③ 또 민다", "④ 자리를 잡았다"]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 5.4), sharex=True)
    rng = np.random.default_rng(1)
    jit = rng.normal(0, 0.05, len(xs))
    for k, (ax, t, cap) in enumerate(zip(axes, thresholds, caps)):
        ax.axhline(0, color="#C9CFD5", lw=1)
        wrong = ((xs > t) & (y == 0)) | ((xs <= t) & (y == 1))
        for cls, c, mk in ((0, UNRIPE, "^"), (1, RIPE, "o")):
            m = (y == cls) & ~wrong
            ax.scatter(xs[m], jit[m], c=c, marker=mk, s=34,
                       edgecolors="white", linewidths=0.5, zorder=3)
            m = (y == cls) & wrong
            ax.scatter(xs[m], jit[m], c="white", marker=mk, s=52,
                       edgecolors=c, linewidths=1.8, zorder=4)
        ax.axvline(t, color=LINE, lw=2.0, zorder=2)
        if k < 3:
            ax.annotate("", xy=(thresholds[k + 1], 0.24), xytext=(t, 0.24),
                        arrowprops=dict(arrowstyle="->", color=ACC, lw=1.6))
        ax.text(0.012, 0.72, cap, transform=ax.transAxes, fontsize=9.5,
                color="#4A4F55")
        n_wrong = int(wrong.sum())
        ax.text(0.988, 0.72, f"틀린 것 {n_wrong}개", transform=ax.transAxes,
                fontsize=9.5, ha="right",
                color=ACC if n_wrong else "#5A9367", fontweight="bold")
        ax.set_ylim(-0.34, 0.40); ax.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("크기 (cm)")
    axes[-1].spines["bottom"].set_visible(True)
    fig.text(0.5, 1.0, "속이 빈 표시가 잘못 분류된 것", ha="center",
             fontsize=9, color="#6B7178")
    fig.tight_layout(h_pad=0.35)
    return save(fig, "fig_2_02_기준이동_4컷")


# ══ 그림 2-3 ══ 1차원에서 2차원으로 ═════════════════════════════════════════
def fig_2_3():
    x, y = data.apples(90, seed=11, noise=0.3)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6),
                                 gridspec_kw={"height_ratios": [1, 3]})
    jit = np.random.default_rng(2).normal(0, 0.04, len(x))
    for cls, c, mk in ((0, UNRIPE, "^"), (1, RIPE, "o")):
        m = y == cls
        a1.scatter(x[m, 0], jit[m], c=c, marker=mk, s=32,
                   edgecolors="white", linewidths=0.5, zorder=3)
    a1.axhline(0, color="#C9CFD5", lw=1)
    a1.set_ylim(-0.3, 0.3); a1.set_yticks([]); a1.set_xlim(0, 4.2)
    a1.set_xlabel("크기")
    a1.set_title("크기만 보면 — 섞여 있다", fontsize=10.5, pad=6)
    for s in ("left", "right", "top"):
        a1.spines[s].set_visible(False)

    scatter(a2, x, y)
    a2.set_xlim(0, 4.2); a2.set_ylim(0, 4.2)
    a2.set_xlabel("크기 $x_1$"); a2.set_ylabel("색깔 $x_2$")
    a2.set_title("색깔 축을 세우면 — 갈린다", fontsize=10.5, pad=6)
    a2.grid(alpha=0.35)
    for k in range(0, len(x), 7):
        a2.annotate("", xy=(x[k, 0], x[k, 1]), xytext=(x[k, 0], -0.15),
                    arrowprops=dict(arrowstyle="-", color="#C9CFD5",
                                    lw=0.7, ls=(0, (2, 2))), zorder=1)
    fig.tight_layout(h_pad=1.4)
    return save(fig, "fig_2_03_1차원에서_2차원으로")


# ══ 그림 2-4 ══ 평면과 가르는 선 ════════════════════════════════════════════
def fig_2_4():
    x, y = data.apples(180, seed=11, noise=0.3)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    shade(ax, 1, 1, -3.5, (0, 4.2), (0, 4.2))
    draw_line(ax, 1, 1, -3.5, (0, 4.2))
    scatter(ax, x, y)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
    ax.set_xlabel("크기 $x_1$"); ax.set_ylabel("색깔 $x_2$")
    ax.text(3.3, 3.6, "익은 것", fontsize=11, color=RIPE, fontweight="bold")
    ax.text(0.35, 0.45, "안 익은 것", fontsize=11, color=UNRIPE, fontweight="bold")
    ax.grid(alpha=0.3)
    return save(fig, "fig_2_04_평면과_가르는선")


# ══ 그림 2-5 ══ 선 위의 점들 ════════════════════════════════════════════════
def fig_2_5():
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    draw_line(ax, 1, 1, -3.5, (0, 4.2))
    for px, py in [(2.5, 1.0), (1.5, 2.0)]:
        ax.scatter([px], [py], s=95, c="white", edgecolors=LINE,
                   linewidths=2, zorder=4)
        ax.annotate(f"({px}, {py})\n{px} + {py} = 3.5",
                    xy=(px, py), xytext=(px + 0.30, py + 0.42),
                    fontsize=10, ha="left",
                    arrowprops=dict(arrowstyle="-", color="#8B9198", lw=0.9))
    ax.text(2.55, 0.30, "$x_1 + x_2 = 3.5$", fontsize=12, color=LINE)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
    ax.set_xlabel("크기 $x_1$"); ax.set_ylabel("색깔 $x_2$")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.grid(alpha=0.4)
    return save(fig, "fig_2_05_선_위의_점들")


# ══ 그림 2-6 ══ 선과 네 점 ══════════════════════════════════════════════════
def fig_2_6():
    pts = [("A", 2, 2, 1), ("B", 3, 1.3, 1), ("C", 1, 1, 0), ("D", 2, 0.7, 0)]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    shade(ax, 1, 1, -3.5, (0, 4.2), (0, 4.2))
    draw_line(ax, 1, 1, -3.5, (0, 4.2))
    for name, px, py, cls in pts:
        c, mk = (RIPE, "o") if cls else (UNRIPE, "^")
        ax.scatter([px], [py], s=110, c=c, marker=mk,
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(f"{name}  {px}+{py} = {px + py:.1f}",
                    xy=(px, py), xytext=(px + 0.16, py + 0.26),
                    fontsize=10, fontweight="bold", color=c)
    ax.text(2.55, 0.28, "$x_1 + x_2 = 3.5$", fontsize=11.5, color=LINE)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
    ax.set_xlabel("크기 $x_1$"); ax.set_ylabel("색깔 $x_2$")
    ax.grid(alpha=0.3)
    return save(fig, "fig_2_06_선과_네점")


# ══ 그림 2-8 ══ 선이 옮겨 간다 ══════════════════════════════════════════════
def fig_2_8():
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    xlim, ylim = (0, 10.5), (0, 4.6)
    shade(ax, 0.1, 0.9, -3.6, xlim, ylim)
    draw_line(ax, 1, 1, -3.5, xlim, color=LINE2, ls="--", lw=1.8,
              label="이전:  $1x_1 + 1x_2 - 3.5 = 0$")
    draw_line(ax, 0.1, 0.9, -3.6, xlim, color=LINE, lw=2.2,
              label="조정 후:  $0.1x_1 + 0.9x_2 - 3.6 = 0$")
    ax.scatter([9], [1], s=170, c=UNRIPE, marker="^",
               edgecolors="white", linewidths=1.4, zorder=5)
    ax.annotate("크기 9, 색깔 1\n실제로는 안 익음",
                xy=(9, 1), xytext=(6.5, 2.3), fontsize=10, color=UNRIPE,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=UNRIPE, lw=1.4))
    ax.text(9.2, 0.42, "이전 선 기준: $+6.5$ → 익음(오판)", fontsize=9,
            ha="right", color=LINE2)
    ax.text(9.2, 0.14, "새 선 기준: $-1.8$ → 안 익음(정답)", fontsize=9,
            ha="right", color=LINE, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("크기 $x_1$"); ax.set_ylabel("색깔 $x_2$")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.28)
    return save(fig, "fig_2_08_선이_옮겨간다")


# ══ 그림 2-9 ══ 학습이 진행되는 6컷 ═════════════════════════════════════════
def fig_2_9():
    x, y = data.apples(400, seed=42)
    s = data.split(x, y, seed=42)
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.6, 2); b = 0.0
    lr, snapshots = 0.02, {}
    want = [0, 1, 3, 6, 14, 30]
    snapshots[0] = (w.copy(), b, metrics.accuracy(s.y_train,
                    (s.x_train @ w + b > 0).astype(int)))
    for ep in range(1, 31):
        for xi, yi in zip(s.x_train, s.y_train):
            pred = 1 if xi @ w + b > 0 else 0
            err = pred - yi
            w -= lr * err * xi; b -= lr * err
        if ep in want:
            snapshots[ep] = (w.copy(), b, metrics.accuracy(
                s.y_train, (s.x_train @ w + b > 0).astype(int)))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    for ax, ep in zip(axes.ravel(), want):
        ww, bb, acc = snapshots[ep]
        shade(ax, ww[0], ww[1], bb, (0, 4.2), (0, 4.2), res=200, alpha=0.55)
        draw_line(ax, ww[0], ww[1], bb, (0, 4.2))
        scatter(ax, s.x_train, s.y_train, s=15, legend=False)
        ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"epoch {ep}   정확도 {acc:.3f}", fontsize=10, pad=4)
    fig.tight_layout()
    return save(fig, "fig_2_09_학습_6컷")


# ══ 그림 2-11 ══ 기준의 위치와 오차 ═════════════════════════════════════════
def fig_2_11():
    x, y = data.tangerines(400, seed=7)
    xs = x[:, 0]
    ts = np.linspace(4.5, 10.5, 300)
    err = [np.mean(((xs > t).astype(int) - y) ** 2) for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ts, err, color=LINE, lw=2.2)
    best = ts[int(np.argmin(err))]
    ax.scatter([best], [min(err)], s=80, c=ACC, zorder=4)
    ax.annotate("여기가 정답\n(오차 최소)", xy=(best, min(err)),
                xytext=(best, max(err) * 0.45), ha="center", fontsize=10,
                color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.3))
    for t, lab in ((5.6, "기준이 왼쪽으로\n밀렸다"), (9.6, "오른쪽으로\n밀렸다")):
        e = np.interp(t, ts, err)
        ax.scatter([t], [e], s=55, c=UNRIPE, zorder=4)
        ax.annotate(lab, xy=(t, e), xytext=(t, e + max(err) * 0.16),
                    ha="center", fontsize=9, color=UNRIPE,
                    arrowprops=dict(arrowstyle="-", color=UNRIPE, lw=0.9))
    ax.set_xlabel("기준의 위치  $w$"); ax.set_ylabel("오차 (손실)")
    ax.grid(alpha=0.32)
    return save(fig, "fig_2_11_기준위치와_오차")


# ══ 그림 2-12 ══ 손실 지형 위의 공 ══════════════════════════════════════════
def fig_2_12():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    g = np.linspace(-3, 3, 90)
    gx, gy = np.meshgrid(g, g)
    zz = 0.35 * (gx ** 2 + gy ** 2) + 1.6 * np.sin(1.1 * gx) * np.cos(1.1 * gy) + 3

    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(gx, gy, zz, cmap="Blues_r", alpha=0.82,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    # 굴러 내려가는 궤적 (수치적 경사하강)
    p = np.array([2.6, 2.3]); path = [p.copy()]
    for _ in range(60):
        h = 1e-4
        gxg = (0.35 * ((p[0] + h) ** 2 + p[1] ** 2)
               + 1.6 * np.sin(1.1 * (p[0] + h)) * np.cos(1.1 * p[1])
               - 0.35 * (p[0] ** 2 + p[1] ** 2)
               - 1.6 * np.sin(1.1 * p[0]) * np.cos(1.1 * p[1])) / h
        gyg = (0.35 * (p[0] ** 2 + (p[1] + h) ** 2)
               + 1.6 * np.sin(1.1 * p[0]) * np.cos(1.1 * (p[1] + h))
               - 0.35 * (p[0] ** 2 + p[1] ** 2)
               - 1.6 * np.sin(1.1 * p[0]) * np.cos(1.1 * p[1])) / h
        p = p - 0.12 * np.array([gxg, gyg]); path.append(p.copy())
    path = np.array(path)
    pz = (0.35 * (path[:, 0] ** 2 + path[:, 1] ** 2)
          + 1.6 * np.sin(1.1 * path[:, 0]) * np.cos(1.1 * path[:, 1]) + 3)
    ax.plot(path[:, 0], path[:, 1], pz + 0.16, color=ACC, lw=2.4, zorder=10)
    ax.scatter(*path[0], pz[0] + 0.25, s=90, c=RIPE, zorder=12)
    ax.scatter(*path[-1], pz[-1] + 0.25, s=90, c=ACC, marker="*", zorder=12)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_zlabel("손실")
    ax.view_init(elev=38, azim=-58)
    ax.set_box_aspect((1, 1, 0.62))
    return save(fig, "fig_2_12_손실지형과_공")


# ══ 그림 2-13 ══ 학습률이 크면 / 작으면 ═════════════════════════════════════
def fig_2_13():
    f = lambda w: 0.5 * w ** 2
    df = lambda w: w
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ax, lr, title in ((axes[0], 1.86, "학습률이 크면\n— 근처에서 방황한다"),
                          (axes[1], 0.16, "학습률이 작으면\n— 도착하지만 오래 걸린다")):
        ws = np.linspace(-3.2, 3.2, 300)
        ax.plot(ws, f(ws), color="#B9C2CA", lw=2.0, zorder=1)
        w, pts = 2.9, [2.9]
        for _ in range(16):
            w = w - lr * df(w); pts.append(w)
        pts = np.array(pts)
        ax.plot(pts, f(pts), "-o", color=ACC, ms=4.6, lw=1.3, zorder=3)
        ax.scatter([pts[0]], [f(pts[0])], s=85, c=RIPE, zorder=5)
        ax.scatter([0], [0], s=105, c=LINE, marker="*", zorder=5)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xlabel("가중치 $w$"); ax.grid(alpha=0.3)
        ax.set_xlim(-3.3, 3.3)
    axes[0].set_ylabel("손실")
    fig.tight_layout()
    return save(fig, "fig_2_13_학습률_크게_작게")


# ══ 그림 2-16 ══ AND · OR · XOR ═════════════════════════════════════════════
def fig_2_16():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tables = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    for ax, (name, lab) in zip(axes, tables.items()):
        lab = np.array(lab)
        for p, l in zip(pts, lab):
            c, mk = (RIPE, "o") if l else (UNRIPE, "^")
            ax.scatter(*p, s=190, c=c, marker=mk,
                       edgecolors="white", linewidths=1.4, zorder=4)
            ax.annotate(str(l), xy=p, xytext=(p[0] + 0.09, p[1] + 0.09),
                        fontsize=11, fontweight="bold", color=c)
        if name == "AND":
            ax.plot([0.2, 1.42], [1.42, 0.2], color=LINE, lw=2.2, zorder=3)
        elif name == "OR":
            ax.plot([-0.35, 0.85], [0.85, -0.35], color=LINE, lw=2.2, zorder=3)
        else:
            for a, b in [(-1.0, 0.5), (-1.0, 1.5), (1.0, 0.0)]:
                xs = np.linspace(-0.35, 1.4, 10)
                ax.plot(xs, a * xs + b, color=LINE2, lw=1.4,
                        ls=(0, (5, 3)), zorder=2)
            ax.set_xlabel("$x_1$\n어떤 직선으로도 안 된다")
        ax.set_xlim(-0.38, 1.42); ax.set_ylim(-0.42, 1.42)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(name, fontsize=12, fontweight="bold", pad=8)
        ax.grid(alpha=0.28)
    fig.tight_layout()
    return save(fig, "fig_2_16_AND_OR_XOR")


# ══ 그림 2-18 ══ 층·노드가 늘면 경계가 복잡해진다 ═══════════════════════════
def fig_2_18():
    rng = np.random.default_rng(4)

    def blobs():
        a = rng.normal([0.7, 0.7], 0.25, (110, 2))
        b = rng.normal([2.3, 2.3], 0.25, (110, 2))
        return np.vstack([a, b]), np.r_[np.zeros(110), np.ones(110)]

    def ring():
        t = rng.uniform(0, 2 * np.pi, 220)
        r = np.r_[rng.uniform(0, 0.7, 110), rng.uniform(1.15, 1.6, 110)]
        p = np.c_[1.5 + r * np.cos(t), 1.5 + r * np.sin(t)]
        return p, np.r_[np.zeros(110), np.ones(110)]

    def spiral():
        n = 110; out = []; lab = []
        for k in range(2):
            t = np.linspace(0.35, 3.1, n)
            r = t * 0.52
            th = t * 2.1 + k * np.pi
            out.append(np.c_[1.5 + r * np.cos(th), 1.5 + r * np.sin(th)]
                       + rng.normal(0, 0.055, (n, 2)))
            lab.append(np.full(n, k))
        return np.vstack(out), np.concatenate(lab)

    def xor4():
        p = rng.uniform(0, 3, (220, 2))
        l = ((p[:, 0] > 1.5) ^ (p[:, 1] > 1.5)).astype(int)
        return p, l

    cases = [("① 직선 하나로", blobs, (0, 1)),
             ("② 꺾인 경계", xor4, (1, 4)),
             ("③ 닫힌 영역", ring, (1, 8)),
             ("④ 복잡한 곡선", spiral, (3, 16))]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4))
    for ax, (title, gen, (L, N)) in zip(axes, cases):
        p, l = gen()
        scatter(ax, p, l, s=13, legend=False)
        ax.set_xlim(-0.15, 3.15); ax.set_ylim(-0.15, 3.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, pad=6)
        cap = ("퍼셉트론 1개 (은닉층 없음)" if L == 0
               else f"은닉층 {L}개 · 노드 {N}개")
        ax.set_xlabel(cap, fontsize=9.5, color="#6B7178")
    fig.tight_layout()
    return save(fig, "fig_2_18_경계_복잡도")




# ══ 그림 9-1 ══ Grad-CAM ════════════════════════════════════════════════════
def fig_9_1():
    """모델이 어디를 보고 판정했는지. 9장 §9.5.

    이 그림은 CNN을 한 번 학습시켜야 하므로 keras가 필요하다.
    keras가 없으면 조용히 건너뛴다.
    """
    try:
        import os
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        import tensorflow as tf
        from keras import layers
        from scipy.ndimage import zoom
    except ImportError:
        print("  (keras/scipy 없음 — fig_9_1 건너뜀)")
        return None

    import dlbook
    dlbook.set_seed(42)
    x, y = data.shapes(6000, seed=42, classes=(0, 1, 2))
    s = data.split(x, y, val_ratio=0.15, test_ratio=0.15, seed=42)

    inp = layers.Input(shape=(28, 28, 1))
    h = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Conv2D(32, 3, activation="relu", padding="same", name="last_conv")(h)
    z = layers.MaxPooling2D(2)(h)
    z = layers.Flatten()(z)
    z = layers.Dense(64, activation="relu")(z)
    out = layers.Dense(3, activation="softmax")(z)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy")
    model.fit(s.x_train, s.y_train, epochs=15, batch_size=64, verbose=0)

    grad_model = keras.Model(model.input,
                             [model.get_layer("last_conv").output, model.output])

    def gradcam(img):
        arr = img[None].astype("float32")
        with tf.GradientTape() as tape:
            conv, pred = grad_model(arr)
            tape.watch(conv)
            cls = tf.argmax(pred[0])
            score = pred[:, cls]
        grads = tape.gradient(score, conv)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(conv[0] * weights, axis=-1).numpy()
        cam = np.maximum(cam, 0)
        return cam / (cam.max() + 1e-8), int(cls)

    names = data.shape_names((0, 1, 2))
    sel = [np.flatnonzero(s.y_test == c)[k] for c in range(3) for k in (0, 1)]
    fig, axes = plt.subplots(2, 6, figsize=(12.2, 4.6),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.06})
    for j, i in enumerate(sel):
        img = s.x_test[i]
        cam, cls = gradcam(img)
        axes[0, j].imshow(img.squeeze(), cmap="gray")
        axes[0, j].set_title(names[int(s.y_test[i])], fontsize=10.5, pad=5)
        axes[1, j].imshow(img.squeeze(), cmap="gray")
        axes[1, j].imshow(zoom(cam, 28 / cam.shape[0], order=1),
                          cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"→ {names[cls]}", fontsize=10.5, pad=5,
                             color="#26282B")
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("원본", fontsize=10.5)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10.5)
    return save(fig, "fig_9_01_gradcam")


# ══ 그림 11-1 ══ 학습된 임베딩 ══════════════════════════════════════════════
def fig_11_1():
    """아무도 가르쳐 주지 않았는데 비슷한 단어가 가까워진다. 11장 §11.3."""
    try:
        import os
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        from keras import layers
    except ImportError:
        print("  (keras 없음 — fig_11_1 건너뜀)")
        return None

    import dlbook
    dlbook.set_seed(42)
    x, y = data.toy_reviews(8000, seed=42)
    s = data.split(x, y, val_ratio=0.2, test_ratio=0.2, seed=42)
    V, L = len(data.TOY_VOCAB), x.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(L,)),
        layers.Embedding(V, 8),
        layers.LSTM(32),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.003, clipnorm=1.0),
                  loss="binary_crossentropy")
    model.fit(s.x_train, s.y_train, epochs=25, batch_size=64, verbose=0)

    E = model.layers[0].get_weights()[0]
    Ec = E - E.mean(0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T

    POS = {"좋다", "훌륭하다", "재미있다", "감동적이다", "만족스럽다"}
    NEG = {"나쁘다", "지루하다", "실망이다", "형편없다", "아쉽다"}

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for i, w in enumerate(data.TOY_VOCAB):
        if w == "<pad>":
            continue
        if w in POS:
            c, sz = RIPE, 70
        elif w in NEG:
            c, sz = UNRIPE, 70
        elif w == "안":
            c, sz = ACC, 90
        else:
            c, sz = "#A9B2BA", 26
        ax.scatter(*P[i], c=c, s=sz, zorder=3, edgecolors="white", linewidths=0.6)
        if w in POS or w in NEG or w == "안":
            ax.annotate(w, P[i], xytext=(6, 4), textcoords="offset points",
                        fontsize=9.5, color=c, fontweight="bold")
    ax.annotate("나머지 단어들\n(내용과 무관)", P[[i for i, w in enumerate(data.TOY_VOCAB)
                                          if w not in POS | NEG and w not in ("<pad>", "안")]].mean(0),
                xytext=(0, 34), textcoords="offset points", ha="center",
                fontsize=9, color="#6B7178")
    ax.set_xlabel("주성분 1"); ax.set_ylabel("주성분 2")
    ax.set_title("학습된 임베딩 — 아무도 가르쳐 주지 않았다", fontsize=11.5, pad=10)
    ax.grid(alpha=0.3)
    return save(fig, "fig_11_01_임베딩")

# ═══════════════════════════════════════════════════════════════════════
#  구조 그림 — 데이터가 아니라 「짜임새」를 보이는 그림들
# ═══════════════════════════════════════════════════════════════════════

BOXC, BOXE = "#F2F5F8", "#8A939C"          # 상자 채움 / 테두리


def _box(ax, x, y, w, h, text, fc=BOXC, ec=BOXE, fs=10, lw=1.2, r=0.06):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)


def _arrow(ax, p, q, color=LINE, lw=1.3, ls="-", shrink=7):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=shrink, shrinkB=shrink,
                                mutation_scale=13))


def _blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    return fig, ax


def fig_1_1():
    """인공지능 ⊃ 머신러닝 ⊃ 딥러닝. 1장 §1.2."""
    from matplotlib.patches import Ellipse
    fig, ax = _blank(6.6, 4.3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6)
    rings = [(5, 3.0, 9.4, 6.0, "#E8EDF2", "#5B6770", "인공지능", 5.55),
             (5, 2.62, 6.9, 4.3, "#DCE8F0", UNRIPE, "머신러닝", 4.34),
             (5, 2.15, 4.0, 2.5, "#F4DAD7", RIPE, "딥러닝", 2.72)]
    for cx, cy, w, h, fc, ec, name, ty in rings:
        ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=1.6, zorder=1))
        ax.text(cx, ty, name, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=ec, zorder=4)
    ax.text(5, 1.72, "신경망을 여러 층", ha="center", fontsize=9, color="#4A4F55")
    ax.text(5, 3.62, "데이터로 규칙을 찾는다", ha="center", fontsize=9, color="#4A4F55")
    ax.text(5, 5.05, "사람이 하던 판단을 기계가", ha="center", fontsize=9, color="#4A4F55")
    ax.text(5, 0.28, "안쪽일수록 좁고 구체적입니다", ha="center",
            fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_1_01_포함관계")


def fig_1_2():
    """신경망은 함수 하나다. 1장 §1.3."""
    fig, ax = _blank(7.6, 2.9)
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
    _box(ax, 1.9, 2.5, 2.9, 1.15, "입력\n(숫자 여러 개)", fc="#DCE8F0", ec=UNRIPE)
    _box(ax, 6.0, 2.5, 3.2, 1.5, "신경망", fc="#F2F5F8", ec=LINE, fs=13)
    _box(ax, 10.1, 2.5, 2.9, 1.15, "출력\n(숫자 여러 개)", fc="#F4DAD7", ec=RIPE)
    _arrow(ax, (3.4, 2.5), (4.4, 2.5)); _arrow(ax, (7.6, 2.5), (8.65, 2.5))
    ax.text(6.0, 1.28, "$y = f(x)$", ha="center", fontsize=13, color=ACC)
    ax.text(6.0, 0.42, "안이 아무리 복잡해도 하는 일은 이것뿐입니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    ax.text(1.9, 3.62, "사과의 무게·색", ha="center", fontsize=8.8, color="#6B7178")
    ax.text(10.1, 3.62, "익었을 확률", ha="center", fontsize=8.8, color="#6B7178")
    return save(fig, "fig_1_02_함수하나")


def _perceptron(ax, cx, cy, labels, act=None, out="y", w_lab=True):
    """퍼셉트론 하나를 그린다. fig_2_7 / 2_14 / 2_15 가 함께 씁니다."""
    n = len(labels)
    ys = np.linspace(cy + 0.95 * (n - 1) / 2, cy - 0.95 * (n - 1) / 2, n)
    for i, (yy, lab) in enumerate(zip(ys, labels)):
        ax.add_patch(plt.Circle((cx - 3.0, yy), 0.30, fc="#DCE8F0",
                                ec=UNRIPE, lw=1.3, zorder=3))
        ax.text(cx - 3.0, yy, lab, ha="center", va="center", fontsize=10, zorder=4)
        _arrow(ax, (cx - 3.0, yy), (cx - 0.62, cy), shrink=10)
        if w_lab:
            t = 0.55
            ax.text(cx - 3.0 + t * 2.38, yy + t * (cy - yy) + 0.20,
                    f"$w_{i+1}$", fontsize=10, color=ACC, ha="center")
    ax.add_patch(plt.Circle((cx, cy), 0.62, fc="#F2F5F8", ec=LINE, lw=1.5, zorder=3))
    ax.text(cx, cy, r"$\Sigma$", ha="center", va="center", fontsize=16, zorder=4)
    if act is None:
        _arrow(ax, (cx + 0.62, cy), (cx + 2.1, cy), shrink=6)
        ax.text(cx + 2.45, cy, out, ha="center", va="center", fontsize=11.5)
    else:
        _arrow(ax, (cx + 0.62, cy), (cx + 1.35, cy), shrink=6)
        _box(ax, cx + 2.05, cy, 1.1, 0.92, act, fc="#FBF0D8", ec=ACC, fs=11)
        _arrow(ax, (cx + 2.6, cy), (cx + 3.5, cy), shrink=6)
        ax.text(cx + 3.85, cy, out, ha="center", va="center", fontsize=11.5)


def fig_2_7():
    """계산의 구조 — 퍼셉트론. 2장 §2.4."""
    fig, ax = _blank(7.4, 3.4)
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    _perceptron(ax, 5.4, 2.6, ["$x_1$", "$x_2$"])
    ax.add_patch(plt.Circle((2.4, 0.85), 0.30, fc="#EEEEEE", ec="#9AA3AB",
                            lw=1.2, ls="--", zorder=3))
    ax.text(2.4, 0.85, "1", ha="center", va="center", fontsize=10, zorder=4)
    _arrow(ax, (2.4, 0.85), (4.78, 2.6), color="#9AA3AB", ls="--", shrink=10)
    ax.text(3.9, 1.36, "$b$", fontsize=10, color=ACC)
    ax.text(5.4, 4.42, r"$z = w_1x_1 + w_2x_2 + b$", ha="center", fontsize=12)
    ax.text(5.4, 0.16, "입력마다 가중치를 곱해 더하고, 편향을 보탭니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_2_07_계산의_구조")


def fig_2_10():
    """가중치의 크기 비교 — 무엇을 더 보고 있는가. 2장 §2.7."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 2.9),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    names = ["무게", "색"]
    for ax, w, ttl in [(a1, [2.8, 0.4], "무게를 주로 봅니다"),
                       (a2, [0.5, 3.1], "색을 주로 봅니다")]:
        ax.barh(names, w, color=[UNRIPE, RIPE], height=0.5, zorder=3)
        for i, v in enumerate(w):
            ax.text(v + 0.12, i, f"{v:.1f}", va="center", fontsize=10.5, color="#4A4F55")
        ax.set_xlim(0, 3.8); ax.set_title(ttl, fontsize=11, pad=8)
        ax.grid(axis="x", alpha=0.35, zorder=0)
        ax.set_xlabel("가중치의 크기", fontsize=9.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("같은 구조인데 가중치가 다르면 다른 것을 봅니다",
                 fontsize=11.5, y=1.04)
    fig.tight_layout()
    return save(fig, "fig_2_10_가중치_크기")


def fig_2_14():
    """신경세포와 퍼셉트론. 2장 §2.12."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.1))
    for ax in (a1, a2):
        ax.set_axis_off(); ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    # 왼쪽 — 신경세포
    a1.add_patch(plt.Circle((5.0, 2.6), 0.95, fc="#F4DAD7", ec=RIPE, lw=1.5, zorder=3))
    a1.text(5.0, 2.6, "세포체", ha="center", va="center", fontsize=10, zorder=4)
    for yy in (3.9, 3.1, 2.3, 1.5):
        a1.plot([1.6, 4.1], [yy, 2.6], color=UNRIPE, lw=1.4, zorder=1)
        a1.plot([1.6], [yy], marker="o", ms=5, color=UNRIPE, zorder=2)
    a1.text(1.5, 4.55, "가지돌기", fontsize=9.5, color=UNRIPE, ha="left")
    a1.plot([5.95, 9.4], [2.6, 2.6], color=LINE, lw=2.0, zorder=1)
    a1.plot([9.4], [2.6], marker=">", ms=8, color=LINE, zorder=2)
    a1.text(7.6, 3.05, "축삭", fontsize=9.5, color=LINE, ha="center")
    a1.text(5.5, 0.5, "신호가 일정 세기를 넘어야 전달됩니다",
            ha="center", fontsize=9, style="italic", color="#6B7178")
    a1.set_title("신경세포", fontsize=12, pad=6)
    # 오른쪽 — 퍼셉트론
    _perceptron(a2, 5.2, 2.6, ["$x_1$", "$x_2$", "$x_3$"], act="계단", w_lab=False)
    a2.text(5.5, 0.5, "합이 문턱을 넘으면 1, 아니면 0",
            ha="center", fontsize=9, style="italic", color="#6B7178")
    a2.set_title("퍼셉트론", fontsize=12, pad=6)
    fig.tight_layout()
    return save(fig, "fig_2_14_신경세포와_퍼셉트론")


def fig_2_15():
    """활성화 함수가 붙은 퍼셉트론. 2장 §2.13."""
    fig, ax = _blank(8.2, 3.2)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    _perceptron(ax, 5.0, 2.6, ["$x_1$", "$x_2$"], act="$a$", w_lab=False)
    ax.text(5.0, 4.45, r"$y = a(w_1x_1 + w_2x_2 + b)$", ha="center", fontsize=12)
    ax.annotate("활성화 함수", xy=(7.05, 3.15), xytext=(8.9, 4.2),
                fontsize=10, color=ACC, ha="center",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=1.1))
    ax.text(5.4, 0.28, "더하기만으로는 직선뿐입니다. 굽히는 것이 여기 붙습니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_2_15_활성화_퍼셉트론")


def fig_2_17():
    """층을 쌓은 구조. 2장 §2.13."""
    fig, ax = _blank(7.8, 3.9)
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.4)
    cols = [(1.6, 2, "입력층", UNRIPE, "#DCE8F0"),
            (4.5, 4, "은닉층", "#5B6770", "#F2F5F8"),
            (7.4, 4, "은닉층", "#5B6770", "#F2F5F8"),
            (10.0, 1, "출력층", RIPE, "#F4DAD7")]
    pos = []
    for cx, n, name, ec, fc in cols:
        ys = np.linspace(3.2 + 0.92 * (n - 1) / 2, 3.2 - 0.92 * (n - 1) / 2, n)
        pos.append(ys)
        for yy in ys:
            ax.add_patch(plt.Circle((cx, yy), 0.29, fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(cx, 5.75, name, ha="center", fontsize=10.5, color=ec)
    for k in range(len(cols) - 1):
        for y1 in pos[k]:
            for y2 in pos[k + 1]:
                ax.plot([cols[k][0] + 0.29, cols[k + 1][0] - 0.29], [y1, y2],
                        color="#C3CAD1", lw=0.75, zorder=1)
    ax.text(5.6, 0.55, "앞 층의 출력이 다음 층의 입력이 됩니다. 그것이 전부입니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_2_17_층을_쌓은_구조")

# ═══════════════════════════════════════════════════════════════════════
#  6~10장 그림
# ═══════════════════════════════════════════════════════════════════════

def fig_6_1():
    """적당한 학습과 과한 학습. 6장 §6.1."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 14))
    y = 0.45 * x + 1.2 + rng.normal(0, 0.7, len(x))
    xs = np.linspace(-0.3, 10.3, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    for ax, deg, ttl in [(a1, 1, "적당한 학습"), (a2, 12, "과한 학습 (과적합)")]:
        c = np.polyfit(x, y, deg)
        ax.plot(xs, np.polyval(c, xs), color=LINE, lw=2, zorder=2)
        ax.scatter(x, y, s=46, color=UNRIPE, ec="white", lw=1.2, zorder=3)
        ax.set_title(ttl, fontsize=11.5, pad=8)
        ax.set_ylim(-1.4, 8.2); ax.set_xlim(-0.5, 10.5)
        ax.grid(alpha=0.3); ax.set_xlabel("입력")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    a1.set_ylabel("출력")
    nx, ny = 5.6, 0.45 * 5.6 + 1.2
    for ax in (a1, a2):
        ax.scatter([nx], [ny], s=95, marker="*", color=ACC, ec="white",
                   lw=1.0, zorder=5)
    c12 = np.polyfit(x, y, 12)
    a2.annotate("새 점은 크게 빗나갑니다",
                xy=(nx, np.polyval(c12, nx)), xytext=(2.0, 7.2), fontsize=9.5,
                color=RIPE, arrowprops=dict(arrowstyle="->", color=RIPE, lw=1.3))
    a1.annotate("새 점", xy=(nx, ny), xytext=(7.0, 0.0), fontsize=9.5,
                color=ACC, arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
    fig.tight_layout()
    return save(fig, "fig_6_01_적당한_학습과_과한_학습")


def fig_6_2():
    """과적합이 시작되는 지점. 6장 §6.1."""
    e = np.arange(1, 61)
    tr = 1.55 * np.exp(-e / 13) + 0.055
    va = 1.55 * np.exp(-e / 11) + 0.30 + 0.0060 * np.clip(e - 22, 0, None) ** 1.18
    k = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(e, tr, color=UNRIPE, lw=2.1, label="학습 손실")
    ax.plot(e, va, color=RIPE, lw=2.1, label="검증 손실")
    ax.axvline(e[k], color=ACC, ls="--", lw=1.4, zorder=1)
    ax.scatter([e[k]], [va[k]], s=70, color=ACC, ec="white", lw=1.2, zorder=4)
    ax.annotate("여기서 멈춰야 합니다", xy=(e[k], va[k]), xytext=(e[k] + 8, va[k] + 0.55),
                fontsize=10, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))
    ax.axvspan(e[k], e[-1], color="#F4DAD7", alpha=0.35, zorder=0)
    ax.text((e[k] + e[-1]) / 2, 1.62, "과적합 구간", ha="center",
            fontsize=10, color=RIPE)
    ax.set_xlabel("epoch"); ax.set_ylabel("손실")
    ax.set_ylim(0, 1.95); ax.set_xlim(1, 60)
    ax.legend(fontsize=10, loc="center right"); ax.grid(alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_6_02_과적합_시작점")


def fig_7_1():
    """ROC 곡선과 AUC. 7장 §7.3."""
    rng = np.random.default_rng(0)
    neg = rng.normal(0.35, 0.16, 900).clip(0, 1)
    pos = rng.normal(0.63, 0.17, 300).clip(0, 1)
    sc = np.concatenate([neg, pos])
    lab = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    th = np.linspace(1, 0, 250)
    tpr = [(sc[lab == 1] >= t).mean() for t in th]
    fpr = [(sc[lab == 0] >= t).mean() for t in th]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.fill_between(fpr, tpr, color=UNRIPE, alpha=0.14, zorder=1)
    ax.plot(fpr, tpr, color=UNRIPE, lw=2.3, zorder=3)
    ax.plot([0, 1], [0, 1], ls="--", color="#9AA3AB", lw=1.3, zorder=2)
    ax.text(0.62, 0.53, "무작위", fontsize=9.5, color="#6B7178", rotation=34)
    ax.text(0.55, 0.28, f"AUC = {auc:.3f}", fontsize=13, color=UNRIPE,
            fontweight="bold")
    ax.annotate("왼쪽 위로 부풀수록\n좋습니다", xy=(0.15, 0.80), xytext=(0.33, 0.93),
                fontsize=9.5, color=ACC, ha="center",
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))
    ax.set_xlabel("거짓 양성 비율 (FPR)"); ax.set_ylabel("참 양성 비율 (TPR)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_7_01_ROC곡선")


def fig_8_1():
    """영상은 숫자 격자다. 8장 §8.1."""
    try:
        img = data.mnist(scale=False).x_test[0].reshape(28, 28)
    except Exception:
        print("  (MNIST 없음 — fig_8_1 건너뜀)"); return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.3),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    a1.imshow(img, cmap="gray_r"); a1.set_title("사람이 보는 것", fontsize=11.5, pad=8)
    a1.set_xticks([]); a1.set_yticks([])
    from matplotlib.patches import Rectangle
    r0, c0, n = 9, 9, 6
    a1.add_patch(Rectangle((c0 - .5, r0 - .5), n, n, fill=False, ec=RIPE, lw=2))
    sub = img[r0:r0 + n, c0:c0 + n]
    a2.imshow(sub, cmap="gray_r", vmin=0, vmax=255)
    for i in range(n):
        for j in range(n):
            v = int(sub[i, j])
            a2.text(j, i, str(v), ha="center", va="center", fontsize=9.5,
                    color="white" if v > 128 else "#26282B")
    a2.set_title("컴퓨터가 보는 것 (0~255)", fontsize=11.5, pad=8)
    a2.set_xticks([]); a2.set_yticks([])
    for s in a2.spines.values(): s.set_edgecolor(RIPE); s.set_linewidth(2)
    fig.suptitle("28 × 28 = 숫자 784개", fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "fig_8_01_영상은_숫자격자")


def fig_8_2():
    """합성곱 연산. 8장 §8.4."""
    rng = np.random.default_rng(7)
    A = rng.integers(0, 10, (6, 6))
    K = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, .18, .55, .18, 1.15])
    ax = fig.add_subplot(gs[0]); axk = fig.add_subplot(gs[2]); axo = fig.add_subplot(gs[4])
    from matplotlib.patches import Rectangle

    def grid(a, M, title, cmap="Blues", hl=None, fs=10):
        a.imshow(M, cmap=cmap, vmin=M.min() - 3, vmax=M.max() + 3)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=fs)
        a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=11, pad=7)
        if hl is not None:
            a.add_patch(Rectangle((hl[1] - .5, hl[0] - .5), 3, 3,
                                  fill=False, ec=RIPE, lw=2.4))

    grid(ax, A, "입력", hl=(1, 1))
    grid(axk, K, "커널 3×3", cmap="Oranges")
    out = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            out[i, j] = (A[i:i+3, j:j+3] * K).sum()
    grid(axo, out, "출력", cmap="Greens", fs=9)
    axo.add_patch(Rectangle((.5, .5), 1, 1, fill=False, ec=RIPE, lw=2.4))
    fig.text(0.365, 0.47, "×", fontsize=20, ha="center", color="#4A4F55")
    fig.text(0.655, 0.47, "→", fontsize=19, ha="center", color="#4A4F55")
    fig.text(0.5, 0.015, "겹친 9칸을 곱해 더한 값 하나. 커널을 한 칸씩 옮기며 되풀이합니다",
             ha="center", fontsize=9.5, style="italic", color="#6B7178")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return save(fig, "fig_8_02_합성곱_연산")


def fig_8_3():
    """최대 풀링. 8장 §8.5."""
    rng = np.random.default_rng(11)
    A = rng.integers(0, 10, (4, 4))
    O = np.array([[A[i:i+2, j:j+2].max() for j in (0, 2)] for i in (0, 2)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    from matplotlib.patches import Rectangle
    a1.imshow(A, cmap="Blues", vmin=-3, vmax=12)
    cols = ["#C1443C", "#B8860B", "#2E6F95", "#2A9D8F"]
    for i in range(4):
        for j in range(4):
            m = (A[(i//2)*2:(i//2)*2+2, (j//2)*2:(j//2)*2+2].max() == A[i, j])
            a1.text(j, i, str(int(A[i, j])), ha="center", va="center",
                    fontsize=13, fontweight="bold" if m else "normal",
                    color=cols[(i//2)*2 + j//2] if m else "#26282B")
    for k, (r, c) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        a1.add_patch(Rectangle((c - .5, r - .5), 2, 2, fill=False,
                               ec=cols[k], lw=2.4))
    a1.set_xticks([]); a1.set_yticks([]); a1.set_title("4×4 입력", fontsize=11.5, pad=8)
    a2.imshow(O, cmap="Greens", vmin=O.min() - 4, vmax=O.max() + 2)
    for i in range(2):
        for j in range(2):
            a2.text(j, i, str(int(O[i, j])), ha="center", va="center",
                    fontsize=17, fontweight="bold", color=cols[i * 2 + j])
    for i in range(2):
        for j in range(2):
            a2.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   ec=cols[i * 2 + j], lw=2.4))
    a2.set_xticks([]); a2.set_yticks([]); a2.set_title("2×2 출력", fontsize=11.5, pad=8)
    fig.text(0.5, 0.02, "구역마다 가장 큰 값만 남깁니다. 크기는 반으로, 위치는 흐릿하게",
             ha="center", fontsize=9.5, style="italic", color="#6B7178")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return save(fig, "fig_8_03_최대_풀링")


def _cnn_stack(ax, cut=False):
    """CNN의 단계별 크기. fig_8_4 와 fig_9_2 가 함께 씁니다."""
    from matplotlib.patches import Rectangle, FancyBboxPatch
    stages = [(28, 1, "28×28×1", "입력"), (28, 16, "28×28×16", "Conv"),
              (14, 16, "14×14×16", "Pool"), (14, 32, "14×14×32", "Conv"),
              (7, 32, "7×7×32", "Pool")]
    x = 0.6
    for k, (s, d, lab, name) in enumerate(stages):
        h = s / 28 * 2.5
        w = 0.20 + d / 32 * 0.62
        ax.add_patch(Rectangle((x, 2.9 - h / 2), w, h, fc="#DCE8F0" if d == 1 else "#CFE0EE",
                               ec=UNRIPE, lw=1.3, zorder=3))
        ax.text(x + w / 2, 2.9 + h / 2 + 0.30, name, ha="center", fontsize=9.5,
                color="#4A4F55")
        ax.text(x + w / 2, 2.9 - h / 2 - 0.38, lab, ha="center", fontsize=8.6,
                color="#6B7178")
        x += w + 0.52
    x += 0.18
    for lab, n, fc, ec in [("1568", 2.0, "#F2F5F8", LINE),
                           ("64", 1.25, "#F2F5F8", LINE),
                           ("3", 0.55, "#F4DAD7", RIPE)]:
        ax.add_patch(FancyBboxPatch((x, 2.9 - n / 2), 0.26, n,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(x + 0.13, 2.9 - n / 2 - 0.38, lab, ha="center", fontsize=8.6,
                color="#6B7178")
        x += 0.26 + 0.50
    ax.text(1.9, 4.75, "특징 추출부", ha="center", fontsize=11.5, color=UNRIPE)
    ax.text(6.15, 4.75, "분류부", ha="center", fontsize=11.5, color=RIPE)
    ax.text(3.05, 1.05, "가로는 줄고 깊이는 늡니다", ha="center", fontsize=9.3,
            style="italic", color="#6B7178")
    if cut:
        ax.plot([4.72, 4.72], [0.75, 4.45], ls="--", color=ACC, lw=2.0, zorder=5)
        ax.text(4.72, 0.42, "여기서 자릅니다", ha="center", fontsize=10,
                color=ACC, fontweight="bold")
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 5.2); ax.set_axis_off()


def fig_8_4():
    """CNN의 구조. 8장 §8.6."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    _cnn_stack(ax)
    fig.tight_layout()
    return save(fig, "fig_8_04_CNN의_구조")


def fig_9_2():
    """CNN을 둘로 자른다. 9장 §9.1."""
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    _cnn_stack(ax, cut=True)
    ax.text(2.2, 0.05, "남이 배운 것을 가져옵니다", ha="center", fontsize=9.5,
            color=UNRIPE)
    ax.text(6.3, 0.05, "내 문제에 맞게 새로 답니다", ha="center", fontsize=9.5,
            color=RIPE)
    fig.tight_layout()
    return save(fig, "fig_9_02_CNN을_둘로")


def fig_10_1():
    """memory_task 가 어떤 문제인가. 10장 §10.2."""
    x, y = data.memory_task(4, length=80, seed=42)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 3.6), sharex=True)
    a1.plot(x[0, :, 0], color=UNRIPE, lw=1.2)
    pos = int(np.argmax(x[0, :, 1]))
    a1.scatter([pos], [x[0, pos, 0]], s=80, color=RIPE, zorder=5, ec="white", lw=1.2)
    a1.annotate("이 값의 부호를 맞혀야 합니다",
                xy=(pos, x[0, pos, 0]), xytext=(pos + 14, x[0, pos, 0] * 0.75),
                fontsize=9.5, color=RIPE,
                arrowprops=dict(arrowstyle="->", color=RIPE, lw=1.2))
    a1.set_ylabel("값", fontsize=10); a1.grid(alpha=0.3)
    a2.plot(x[0, :, 1], color=ACC, lw=1.2)
    a2.set_ylabel("표시", fontsize=10); a2.set_xlabel("걸음 (0 ~ 79)")
    a2.grid(alpha=0.3); a2.set_ylim(-0.15, 1.2)
    for ax in (a1, a2):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    a2.annotate("", xy=(79, 0.55), xytext=(pos, 0.55),
                arrowprops=dict(arrowstyle="<->", color="#9AA3AB", lw=1.1))
    a2.text((pos + 79) / 2, 0.72, f"{79 - pos}걸음을 들고 가야 합니다",
            ha="center", fontsize=9.3, color="#6B7178")
    fig.suptitle("대부분은 잡음입니다. 한 자리만 답과 관계있습니다", fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "fig_10_01_memory_task")


def fig_10_2():
    """순환 구조를 펼치면. 10장 §10.3."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.2),
                                 gridspec_kw={"width_ratios": [1, 2.7]})
    for ax in (a1, a2):
        ax.set_axis_off()
    # 왼쪽 — 접힌 모습
    a1.set_xlim(0, 4); a1.set_ylim(0, 4.6)
    _box(a1, 2.0, 2.3, 1.5, 1.1, "RNN", fc="#F2F5F8", ec=LINE, fs=12)
    a1.add_patch(FancyArrowPatch((2.75, 2.75), (2.75, 1.85),
                                 connectionstyle="arc3,rad=-1.9",
                                 arrowstyle="-|>", color=ACC, lw=1.6,
                                 mutation_scale=13))
    a1.text(3.62, 2.3, "$h$", fontsize=12, color=ACC, va="center")
    _arrow(a1, (2.0, 0.85), (2.0, 1.72), shrink=4)
    a1.text(2.0, 0.5, "$x_t$", ha="center", fontsize=11)
    _arrow(a1, (2.0, 2.9), (2.0, 3.8), shrink=4)
    a1.text(2.0, 4.12, "$y_t$", ha="center", fontsize=11)
    a1.set_title("접어서 보면", fontsize=11.5, pad=6)
    # 오른쪽 — 펼친 모습
    a2.set_xlim(0, 12); a2.set_ylim(0, 4.6)
    xs = [1.6, 4.2, 6.8, 10.0]
    labs = ["$x_1$", "$x_2$", "$x_3$", "$x_T$"]
    for k, (cx, lb) in enumerate(zip(xs, labs)):
        _box(a2, cx, 2.3, 1.35, 1.05, "RNN", fc="#F2F5F8", ec=LINE, fs=10.5)
        _arrow(a2, (cx, 0.95), (cx, 1.72), shrink=4)
        a2.text(cx, 0.58, lb, ha="center", fontsize=10.5)
        if k < len(xs) - 1:
            nx = xs[k + 1]
            if k == 2:
                a2.text((cx + nx) / 2, 2.3, "· · ·", ha="center", va="center",
                        fontsize=14, color="#9AA3AB")
            else:
                _arrow(a2, (cx + 0.68, 2.3), (nx - 0.68, 2.3), color=ACC, shrink=2)
                a2.text((cx + nx) / 2, 2.62, "$h$", ha="center", fontsize=10.5, color=ACC)
    _arrow(a2, (10.0, 2.85), (10.0, 3.7), shrink=4)
    a2.text(10.0, 4.02, "$y_T$", ha="center", fontsize=10.5)
    a2.text(6.0, 0.03, "상자가 여럿으로 보이지만 같은 가중치 하나입니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    a2.set_title("시간축으로 펼치면", fontsize=11.5, pad=6)
    fig.tight_layout()
    return save(fig, "fig_10_02_순환구조_펼치기")

# ═══════════════════════════════════════════════════════════════════════
#  5 · 12~14장 그림 — 수치는 figures/measured.json 에서 읽습니다
#
#  이 파일은 노트북을 실제로 돌려 얻은 값을 모아 둔 것입니다.
#  본문의 표와 같은 값이므로 그림과 표가 어긋날 수 없습니다.
#
#  예전 이름은 expected.json 이었고, CI가 매주 노트북을 돌려 이 값과
#  대조했습니다. 그 대조는 폐기했습니다 — 딥러닝은 돌릴 때마다 값이 달라서
#  가짜 경보만 냈고, 이제는 노트북에 실행 결과를 담아 커밋하므로
#  본문 수치가 맞는지는 노트북을 열어 보면 됩니다.
#  남은 쓸모는 하나, **아래 다섯 그림을 그리는 데이터**입니다.
# ═══════════════════════════════════════════════════════════════════════

def _exp():
    """measured.json 을 읽어 판 접두어를 벗긴 사전으로 돌려준다."""
    import json
    p = ROOT / "measured.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 가 없습니다. 이 파일 없이는 5·12·13·14장 그림을 그릴 수 없습니다."
        )
    raw = json.load(open(p, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        name = k.split("::")[-1]
        if "::" not in k or k.startswith("keras::"):
            out[name] = v
        else:
            out.setdefault(name, v)
    return out


def fig_3_1():
    """Keras 3는 TF와 나란한 것이 아니다. 3장 §3.3."""
    fig, ax = _blank(7.6, 3.5)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)
    _box(ax, 6.0, 4.35, 5.2, 1.05, "Keras 3  —  상위 API",
         fc="#F4DAD7", ec=RIPE, fs=12.5)
    for cx, name in [(2.3, "TensorFlow"), (6.0, "PyTorch"), (9.7, "JAX")]:
        _box(ax, cx, 2.15, 3.0, 1.05, name, fc="#DCE8F0", ec=UNRIPE, fs=11.5)
        _arrow(ax, (cx, 2.72), (cx if cx == 6.0 else 6.0 + (cx - 6.0) * 0.34, 3.78),
               color="#8A939C", shrink=4)
    ax.text(6.0, 1.05, "엔진 — 실제 계산은 이 셋 중 하나가 합니다",
            ha="center", fontsize=10, color="#4A4F55")
    ax.text(6.0, 0.35, "같은 Keras 코드가 엔진만 갈아 끼워 그대로 돕니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_3_01_keras3_구조")


def fig_4_1():
    """개수를 먼저 정합니다. 4장 §4.2."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    sets = [("400개", 200, 100, 100), ("1,000개", 800, 100, 100),
            ("10,000개", 9000, 500, 500), ("1,000,000개", 980000, 10000, 10000)]
    y = np.arange(len(sets))[::-1]
    for i, (name, tr, va, te) in zip(y, sets):
        tot = tr + va + te
        l = 0
        for v, c, lab in ((tr, UNRIPE, "train"), (va, ACC, "val"), (te, RIPE, "test")):
            ax.barh(i, v / tot, left=l / tot, color=c, height=0.55,
                    ec="white", lw=1.2, zorder=3)
            if v / tot > 0.055:
                ax.text(l / tot + v / tot / 2, i, f"{v:,}", ha="center", va="center",
                        color="white", fontsize=9.5, fontweight="bold")
            l += v
        ax.text(1.015, i, f"train {tr/tot*100:.0f}%", va="center",
                fontsize=9.5, color="#4A4F55")
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in sets], fontsize=10.5)
    ax.set_xlim(0, 1.16); ax.set_xticks([])
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=UNRIPE, label="train"),
                       Patch(color=ACC, label="val (≥100)"),
                       Patch(color=RIPE, label="test (≥100)")],
              loc="lower center", bbox_to_anchor=(0.42, -0.30), ncol=3, fontsize=9.5,
              frameon=False)
    ax.set_title("val·test에 최소 개수를 먼저 떼고, 남은 것을 전부 학습에",
                 fontsize=11.5, pad=10)
    fig.tight_layout()
    return save(fig, "fig_4_01_개수가_먼저")


def fig_5_1():
    """학습률 하나로 결론이 뒤집힙니다. 5장 §5.4."""
    e = _exp()
    lrs = [0.0003, 0.001, 0.003, 0.01, 0.03]
    acts = [("relu", "ReLU", RIPE, "o"), ("tanh", "tanh", "#2A9D8F", "s"),
            ("sigmoid", "시그모이드", UNRIPE, "^")]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for key, lab, c, m in acts:
        ys = [e.get(f"ch05_d10_{key}_lr{lr}_acc") for lr in lrs]
        if any(v is None for v in ys): continue
        ax.plot(range(len(lrs)), ys, marker=m, ms=7, lw=2.0, color=c, label=lab)
    ax.axhline(0.5, ls="--", color="#9AA3AB", lw=1.2)
    ax.text(0.06, 0.517, "동전 던지기", fontsize=9, color="#6B7178")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([str(l) for l in lrs])
    ax.set_xlabel("학습률"); ax.set_ylabel("시험 정확도")
    ax.set_ylim(0.42, 1.06); ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="center left")
    ax.set_title("10층에서 활성화 함수별 — 학습률을 훑어야 보입니다",
                 fontsize=11.5, pad=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_5_01_학습률_훑기")


def fig_5_2():
    """깊이의 벽은 학습률이었습니다. 5장 §5.4."""
    e = _exp()
    depths, lrs = [10, 20, 30], [0.001, 0.01]
    M = np.array([[e.get(f"ch05_relu_d{d}_lr{lr}_acc", np.nan) for lr in lrs]
                  for d in depths])
    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.45, vmax=1.0, aspect="auto")
    for i in range(len(depths)):
        for j in range(len(lrs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="#26282B" if M[i, j] > 0.7 else "white")
    ax.set_xticks(range(len(lrs))); ax.set_xticklabels([f"lr={l}" for l in lrs], fontsize=11)
    ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d}층" for d in depths], fontsize=11)
    ax.set_title("ReLU — 깊이가 아니라 학습률이 갈랐습니다", fontsize=11.5, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="시험 정확도")
    fig.tight_layout()
    return save(fig, "fig_5_02_깊이와_학습률")


def fig_12_1():
    """어텐션 — 점수 → 소프트맥스 → 가중합. 12장 §12.2."""
    toks = ["영화", "안", "좋다", "정말", "배우", "나쁘다"]
    score = np.array([0.2, 3.1, 1.4, 0.1, 0.0, 0.1])
    w = np.exp(score) / np.exp(score).sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    a1.bar(range(len(toks)), score, color="#C3CAD1", ec="#8A939C", zorder=3)
    a1.bar([1], [score[1]], color=ACC, ec=ACC, zorder=4)
    a1.set_xticks(range(len(toks))); a1.set_xticklabels(toks, fontsize=10)
    a1.set_ylabel("관련 점수"); a1.grid(axis="y", alpha=0.3)
    a1.set_title("① 얼마나 관련 있는가", fontsize=11, pad=8)
    a2.bar(range(len(toks)), w, color="#CFE0EE", ec=UNRIPE, zorder=3)
    a2.bar([1], [w[1]], color=RIPE, ec=RIPE, zorder=4)
    for i, v in enumerate(w):
        a2.text(i, v + 0.022, f"{v:.2f}", ha="center", fontsize=9.5,
                color=RIPE if i == 1 else "#4A4F55")
    a2.set_xticks(range(len(toks))); a2.set_xticklabels(toks, fontsize=10)
    a2.set_ylabel("가중치 (합 = 1)"); a2.set_ylim(0, 0.88); a2.grid(axis="y", alpha=0.3)
    a2.set_title("② 소프트맥스 → ③ 이 비율로 섞는다", fontsize=11, pad=8)
    for ax in (a1, a2):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle('"좋다"가 문장을 볼 때 — 앞에 붙은 "안"을 71% 가져옵니다',
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    return save(fig, "fig_12_01_어텐션_세단계")


def fig_12_2():
    """위치 인코딩 — 자리마다 다른 무늬. 12장 §12.5."""
    L, D = 16, 8
    pos = np.arange(L)[:, None]; i = np.arange(D)[None, :]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / D)
    PE = np.zeros((L, D)); PE[:, 0::2] = np.sin(ang[:, 0::2]); PE[:, 1::2] = np.cos(ang[:, 1::2])
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    im = ax.imshow(PE.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("자리 (0 ~ 15)"); ax.set_ylabel("차원 (0 ~ 7)")
    ax.set_xticks(range(0, L, 2)); ax.set_yticks(range(D))
    ax.set_title("세로줄 하나가 자리 하나 — 자리마다 무늬가 다릅니다",
                 fontsize=11.5, pad=9)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    return save(fig, "fig_12_02_위치_인코딩")


def fig_12_3():
    """길이가 늘면 LSTM이 무너집니다. 12장 §12.6."""
    e = _exp()
    L = [16, 32, 64]
    series = [("cnn", "Conv1D", "#2A9D8F", "s"), ("lstm", "LSTM", UNRIPE, "o"),
              ("attn_pos", "어텐션 + 위치 인코딩", RIPE, "D")]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    for key, lab, c, m in series:
        ys = [e.get(f"ch12_L{l}_{key}_best") for l in L]
        if any(v is None for v in ys): continue
        ax.plot(L, ys, marker=m, ms=8, lw=2.2, color=c, label=lab)
    ax.axhline(0.5, ls="--", color="#9AA3AB", lw=1.2)
    ax.text(17, 0.517, "동전 던지기", fontsize=9, color="#6B7178")
    ax.annotate("학습률 3종 어디에서도\n0.499", xy=(64, 0.499), xytext=(40, 0.65),
                fontsize=9.5, color=UNRIPE, ha="center",
                arrowprops=dict(arrowstyle="->", color=UNRIPE, lw=1.3))
    ax.set_xticks(L); ax.set_xlabel("문장 길이"); ax.set_ylabel("시험 정확도")
    ax.set_ylim(0.42, 1.06); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="center left")
    ax.set_title("각 점은 학습률 3종 중 최고입니다", fontsize=11.5, pad=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_12_03_길이와_구조")


def fig_12_4():
    """트랜스포머 블록. 12장 §12.7."""
    fig, ax = _blank(5.4, 5.4)
    ax.set_xlim(0, 8); ax.set_ylim(0, 11)
    _box(ax, 4, 0.7, 2.4, 0.8, "입력", fc="#DCE8F0", ec=UNRIPE, fs=11)
    _box(ax, 4, 3.0, 4.4, 1.0, "① 멀티헤드 셀프 어텐션", fc="#F4DAD7", ec=RIPE, fs=10.5)
    _box(ax, 4, 4.7, 4.4, 0.8, "② 잔차 + ③ 층 정규화", fc="#FBF0D8", ec=ACC, fs=10)
    _box(ax, 4, 7.0, 4.4, 1.0, "④ 위치별 완전연결층", fc="#E8EDF2", ec="#5B6770", fs=10.5)
    _box(ax, 4, 8.7, 4.4, 0.8, "② 잔차 + ③ 층 정규화", fc="#FBF0D8", ec=ACC, fs=10)
    _box(ax, 4, 10.4, 2.4, 0.8, "출력", fc="#F4DAD7", ec=RIPE, fs=11)
    for a, b in [(1.1, 2.5), (3.5, 4.3), (5.1, 6.5), (7.5, 8.3), (9.1, 10.0)]:
        _arrow(ax, (4, a), (4, b), shrink=2)
    for y0, y1 in [(1.1, 4.5), (5.3, 8.5)]:
        ax.plot([1.5, 1.5], [y0, y1], color=ACC, lw=1.4, ls="--", zorder=1)
        ax.plot([1.5, 4 - 2.2], [y0, y0], color=ACC, lw=1.4, ls="--", zorder=1)
        _arrow(ax, (1.5, y1), (4 - 2.25, y1), color=ACC, lw=1.4, shrink=1)
    ax.text(0.75, 6.6, "잔차 연결\n(9장)", fontsize=9, color=ACC, ha="center", rotation=90)
    ax.text(4, 0.03, "새로운 것은 ① 하나뿐입니다", ha="center",
            fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_12_04_트랜스포머_블록")


def fig_13_1():
    """오토인코더 — 가운데가 좁다. 13장 §13.1."""
    fig, ax = _blank(8.0, 3.2)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon([[2.4, 4.3], [4.9, 3.15], [4.9, 1.85], [2.4, 0.7]],
                         fc="#DCE8F0", ec=UNRIPE, lw=1.4, zorder=2))
    ax.add_patch(Polygon([[7.1, 3.15], [9.6, 4.3], [9.6, 0.7], [7.1, 1.85]],
                         fc="#F4DAD7", ec=RIPE, lw=1.4, zorder=2))
    _box(ax, 6.0, 2.5, 1.5, 1.35, "32", fc="#FBF0D8", ec=ACC, fs=13)
    ax.text(3.65, 2.5, "인코더", ha="center", va="center", fontsize=11, color=UNRIPE)
    ax.text(8.35, 2.5, "디코더", ha="center", va="center", fontsize=11, color=RIPE)
    _box(ax, 1.0, 2.5, 1.5, 2.9, "784", fc="#F2F5F8", ec=LINE, fs=12)
    _box(ax, 11.0, 2.5, 1.5, 2.9, "784", fc="#F2F5F8", ec=LINE, fs=12)
    ax.text(6.0, 0.95, "병목", ha="center", fontsize=10, color=ACC)
    ax.annotate("", xy=(11.0, 4.35), xytext=(1.0, 4.35),
                arrowprops=dict(arrowstyle="<->", color="#8A939C", lw=1.2,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(6.0, 4.92, "이 둘이 같아지도록 학습합니다  —  정답이 곧 입력",
            ha="center", fontsize=10, color="#4A4F55")
    ax.text(6.0, 0.12, "좁은 곳을 지나가게 해야 「무엇이 중요한지」를 고르게 됩니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_13_01_오토인코더_구조")


def fig_13_2():
    """오토인코더 대 PCA. 13장 §13.3~13.4."""
    e = _exp()
    L = [2, 8, 32, 64]
    ae = [e.get(f"ch13_ae_mse_{k}") for k in L]
    pca = [e.get(f"ch13_pca_mse_{k}") for k in L]
    lin = {k: e.get(f"ch13_linear_mse_{k}") for k in (2, 32)}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(L))
    ax.bar(x - 0.19, pca, 0.36, label="PCA", color="#C3CAD1", ec="#8A939C", zorder=3)
    ax.bar(x + 0.19, ae, 0.36, label="오토인코더 (비선형)", color=UNRIPE,
           ec=UNRIPE, zorder=3)
    for k, (i, d) in enumerate(zip(x, L)):
        if lin.get(d) is not None:
            ax.plot([i - 0.19], [lin[d]], marker="_", ms=26, mew=3.0,
                    color=RIPE, zorder=6,
                    label="선형 오토인코더" if k == 0 else None)
    for i, (a, p) in enumerate(zip(ae, pca)):
        ax.text(i + 0.19, a + 0.0016, f"{a:.4f}", ha="center", fontsize=8.6, color=UNRIPE)
        ax.text(i - 0.19, p + 0.0016, f"{p:.4f}", ha="center", fontsize=8.6, color="#6B7178")
    ax.set_xticks(x); ax.set_xticklabels([f"잠재 {d}" for d in L], fontsize=10.5)
    ax.set_ylabel("복원 MSE (낮을수록 좋다)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9.5)
    ax.set_title("이기는 이유는 깊이가 아니라 비선형입니다\n"
                 "— 활성화를 빼면(붉은 선) PCA와 겹칩니다", fontsize=11, pad=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_13_02_AE와_PCA")


def fig_14_1():
    """GAN — 두 모델이 서로를 이기려 합니다. 14장 §14.2."""
    fig, ax = _blank(8.4, 3.6)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)
    _box(ax, 1.3, 3.4, 1.7, 0.85, "잡음 $z$", fc="#F2F5F8", ec=LINE, fs=10.5)
    _box(ax, 4.2, 3.4, 2.5, 1.1, "생성기 G", fc="#F4DAD7", ec=RIPE, fs=12)
    _box(ax, 4.2, 1.15, 2.5, 0.85, "진짜 그림", fc="#DCE8F0", ec=UNRIPE, fs=10.5)
    _box(ax, 8.0, 2.3, 2.5, 1.1, "판별기 D", fc="#DCE8F0", ec=UNRIPE, fs=12)
    _box(ax, 11.0, 2.3, 1.8, 0.85, "진짜? 가짜?", fc="#FBF0D8", ec=ACC, fs=10)
    _arrow(ax, (2.15, 3.4), (2.95, 3.4), shrink=3)
    _arrow(ax, (5.45, 3.4), (6.9, 2.72), shrink=3)
    _arrow(ax, (5.45, 1.15), (6.9, 1.95), shrink=3)
    _arrow(ax, (9.25, 2.3), (10.1, 2.3), shrink=3)
    ax.text(6.2, 3.35, "가짜", fontsize=9.5, color=RIPE)
    ax.text(6.2, 1.32, "진짜", fontsize=9.5, color=UNRIPE)
    ax.annotate("", xy=(4.2, 4.05), xytext=(8.0, 4.05),
                arrowprops=dict(arrowstyle="->", color=RIPE, lw=1.6,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(6.1, 5.05, "G는 D를 속이려 하고", ha="center", fontsize=10, color=RIPE)
    ax.text(8.0, 0.75, "D는 안 속으려 합니다", ha="center", fontsize=10, color=UNRIPE)
    ax.text(6.0, 0.08, "아무도 「이 그림이 정답」이라고 말해 주지 않습니다",
            ha="center", fontsize=9.5, style="italic", color="#6B7178")
    return save(fig, "fig_14_01_GAN_구조")


def fig_14_2():
    """손실이 오르는데 결과는 좋아집니다. 14장 §14.3."""
    e = _exp()
    ep = [1, 2, 5, 10, 20, 30]
    gl = [e.get(f"ch14_gloss_epoch{k}") for k in ep]
    dv = [e.get(f"ch14_div_epoch{k}") for k in ep]
    real = e.get("ch14_real_diversity")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(ep, gl, marker="o", ms=7, lw=2.2, color=RIPE, label="G 손실")
    ax.set_xlabel("epoch"); ax.set_ylabel("G 손실", color=RIPE)
    ax.tick_params(axis="y", labelcolor=RIPE)
    ax.set_ylim(0, max(gl) * 1.25); ax.grid(alpha=0.3)
    a2 = ax.twinx()
    a2.plot(ep, dv, marker="s", ms=7, lw=2.2, color=UNRIPE, label="다양성")
    if real: a2.axhline(real, ls="--", color="#9AA3AB", lw=1.3)
    a2.text(2, real * 1.02, f"진짜 데이터 {real:.1f}", fontsize=9, color="#6B7178")
    a2.set_ylabel("생성물끼리 평균 거리", color=UNRIPE)
    a2.tick_params(axis="y", labelcolor=UNRIPE)
    a2.set_ylim(0, real * 1.28)
    ax.annotate("손실은 오르는데", xy=(20, gl[4]), xytext=(11, gl[4] * 1.28),
                fontsize=10, color=RIPE,
                arrowprops=dict(arrowstyle="->", color=RIPE, lw=1.2))
    a2.annotate("결과는 좋아집니다", xy=(20, dv[4]), xytext=(9.5, dv[4] * 0.66),
                fontsize=10, color=UNRIPE,
                arrowprops=dict(arrowstyle="->", color=UNRIPE, lw=1.2))
    ax.set_title("GAN에서 손실은 성능 지표가 아닙니다", fontsize=11.5, pad=9)
    for s in ("top",): ax.spines[s].set_visible(False); a2.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_14_02_손실과_다양성")


def fig_15_1():
    """네 구조를 한 장에 — 데이터의 성질이 구조를 정한다."""
    def frame(ax, title, data_line, assume):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0.012, 0.012), 0.976, 0.976, fill=False,
                               ec="#DFE3E7", lw=1.0))
        ax.text(0.5, 0.945, title, ha="center", va="top",
                fontsize=11.5, fontweight="bold", color=LINE)
        ax.text(0.5, 0.862, data_line, ha="center", va="top",
                fontsize=9.2, color="#4A4F55")
        ax.plot([0.30, 0.70], [0.185, 0.185], color="#DFE3E7", lw=0.9)
        ax.text(0.5, 0.105, assume, ha="center", va="center",
                fontsize=9.8, color=ACC, fontweight="bold")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
    fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.008, right=0.992,
                        top=0.992, bottom=0.008)

    # ── ① 완전연결 ──────────────────────────────────────────────
    ax = axes[0][0]
    frame(ax, "① 완전연결 — DNN", "값이 나열되어 있을 뿐이다",
          "가정하는 것이 없다")
    xs = np.linspace(0.22, 0.78, 4); hs = np.linspace(0.31, 0.69, 3)
    yi, yh = 0.62, 0.34
    for x in xs:
        for h in hs:
            ax.plot([x, h], [yi, yh], color=LINE2, lw=0.55, zorder=1)
    ax.scatter(xs, [yi]*4, s=190, facecolor=FILL_U, edgecolor=UNRIPE,
               linewidth=1.3, zorder=3)
    ax.scatter(hs, [yh]*3, s=190, facecolor="white", edgecolor=LINE,
               linewidth=1.3, zorder=3)
    ax.text(0.5, 0.735, "모든 입력이 모든 뉴런으로", ha="center",
            fontsize=9.0, color=LINE2)
    ax.text(0.5, 0.245, "연결이 가장 많고, 아는 것이 가장 적다",
            ha="center", fontsize=8.9, color="#4A4F55")

    # ── ② CNN ──────────────────────────────────────────────────
    ax = axes[0][1]
    frame(ax, "② 합성곱 — CNN", "격자다. 이웃한 칸이 서로 관계있다",
          "가까운 것끼리 관계있다")
    c = 0.062; x0 = 0.16; y0 = 0.36
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((x0 + j*c, y0 + i*c), c, c, fc="white",
                                   ec="#DFE3E7", lw=0.8))
    ax.add_patch(Rectangle((x0, y0 + 2*c), 3*c, 3*c, fc=FILL_U, ec=UNRIPE,
                           lw=1.7, alpha=0.9, zorder=2))
    ax.add_patch(Rectangle((x0 + 2*c, y0), 3*c, 3*c, fill=False, ec=UNRIPE,
                           lw=1.2, ls=(0, (3, 2)), zorder=2))
    ax.text(x0 + 1.5*c, y0 + 5*c + 0.035, "커널", ha="center",
            fontsize=8.4, color=UNRIPE)
    ax.add_patch(Rectangle((0.70, y0 + 3.3*c), c, c, fc=FILL_R, ec=RIPE,
                           lw=1.4, zorder=3))
    ax.add_patch(FancyArrowPatch((x0 + 3*c + 0.012, y0 + 3.5*c),
                                 (0.695, y0 + 3.8*c),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=LINE, lw=1.1, zorder=3))
    ax.text(0.735, y0 + 2.55*c, "값 하나", ha="center", fontsize=8.5, color=LINE)
    ax.text(0.5, 0.245, "같은 커널을 옮겨 쓴다 — 파라미터 공유",
            ha="center", fontsize=8.9, color="#4A4F55")

    # ── ③ RNN·LSTM ─────────────────────────────────────────────
    ax = axes[1][0]
    frame(ax, "③ 순환 — RNN · LSTM", "순서가 있다. 앞이 뒤에 영향을 준다",
          "한 칸씩 차례로 지나간다")
    bx = np.linspace(0.14, 0.73, 4); by, bw, bh = 0.50, 0.13, 0.155
    for k, x in enumerate(bx):
        ax.add_patch(Rectangle((x, by), bw, bh, fc=FILL_U, ec=UNRIPE, lw=1.2))
        ax.text(x + bw/2, by + bh/2, f"$x_{k+1}$", ha="center", va="center",
                fontsize=9.5, color=LINE)
        if k < 3:
            ax.add_patch(FancyArrowPatch((x + bw, by + bh/2), (bx[k+1], by + bh/2),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=LINE, lw=1.2))
            ax.text((x + bw + bx[k+1])/2, by + bh/2 + 0.042, "상태",
                    ha="center", fontsize=7.6, color=LINE2)
    ax.add_patch(FancyArrowPatch((bx[0] + bw/2, by - 0.015),
                                 (bx[3] + bw/2, by - 0.015),
                                 connectionstyle="arc3,rad=0.28",
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=RIPE, lw=1.2, ls=(0, (3, 2))))
    ax.text(0.5, 0.232, "$x_1$이 $x_4$에 닿으려면 세 칸을 거친다",
            ha="center", fontsize=8.9, color=RIPE)

    # ── ④ 트랜스포머 ────────────────────────────────────────────
    ax = axes[1][1]
    frame(ax, "④ 어텐션 — 트랜스포머", "순서가 있다. 그런데 한꺼번에 본다",
          "모두가 모두를 직접 본다")
    bx = np.linspace(0.14, 0.73, 4); by, bw, bh = 0.44, 0.13, 0.155
    src = 3
    for k, x in enumerate(bx):
        ax.add_patch(Rectangle((x, by), bw, bh, fc=FILL_R if k == src else FILL_U,
                               ec=RIPE if k == src else UNRIPE, lw=1.2, zorder=3))
        ax.text(x + bw/2, by + bh/2, f"$x_{k+1}$", ha="center", va="center",
                fontsize=9.5, color=LINE, zorder=4)
    w = [0.6, 2.4, 1.2, 0]
    for k, x in enumerate(bx):
        if k == src:
            continue
        ax.add_patch(FancyArrowPatch((bx[src] + bw/2, by + bh),
                                     (x + bw/2, by + bh),
                                     connectionstyle="arc3,rad=0.40",
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=RIPE, lw=w[k], alpha=0.85, zorder=2))
    ax.text(0.5, 0.335, "굵기 = 어텐션 가중치", ha="center",
            fontsize=8.4, color=RIPE)
    ax.text(0.5, 0.245, "거리가 얼마든 한 걸음이다",
            ha="center", fontsize=8.9, color="#4A4F55")

    return save(fig, "fig_15_01_네_구조_한_장")


if __name__ == "__main__":
    print("그림 생성:")
    for fn in (fig_2_1, fig_2_2, fig_2_3, fig_2_4, fig_2_5, fig_2_6,
               fig_2_8, fig_2_9, fig_2_11, fig_2_12, fig_2_13,
               fig_2_16, fig_2_18, fig_9_1, fig_11_1, fig_15_1):
        fn()
    print("완료")
