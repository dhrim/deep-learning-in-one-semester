"""그림 — matplotlib만 쓴다. 세 판의 노트북이 같은 셀을 공유한다.

한글 폰트는 plot.use_korean()이 실행 환경에서 알아서 찾는다.
Kaggle·Colab·JupyterHub 어디서 열어도 축 이름이 네모로 깨지지 않도록 하는 것이
이 모듈의 숨은 목적이다. (기존 노트북에서 가장 흔했던 사고다.)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "use_korean",
    "has_korean",
    "loss_curve",
    "confusion",
    "image_grid",
    "scatter2d",
    "decision_boundary",
    "roc",
]

_KOREAN_CANDIDATES = (
    "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
    "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum",
)


_HAS_KOREAN: bool | None = None

# 한글 폰트가 없는 환경에서 이 모듈이 만드는 축 이름만 영문으로 바꾼다.
# 학생이 직접 쓴 한글 문자열까지 번역하지는 않는다 — 그건 학생의 선택이다.
_KO2EN = {
    "크기": "size", "색깔": "color", "학습 곡선": "learning curve",
    "혼동행렬": "confusion matrix", "예측": "predicted", "실제": "actual",
    "안 익음": "unripe", "익음": "ripe", "정답": "true",
    "ROC 곡선": "ROC curve", "무작위": "random",
    "거짓 양성 비율 (FPR)": "false positive rate (FPR)",
    "참 양성 비율 (TPR)": "true positive rate (TPR)",
}


def _L(text):
    """한글 폰트가 없으면 이 모듈이 쓰는 라벨을 영문으로 바꿔 준다."""
    if text is None or has_korean():
        return text
    return _KO2EN.get(text, text)


def has_korean() -> bool:
    """한글 폰트를 쓸 수 있는 환경인가. 처음 물으면 use_korean()을 부른다."""
    if _HAS_KOREAN is None:
        use_korean()
    return bool(_HAS_KOREAN)


def use_korean(verbose: bool = False) -> str | None:
    """설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 지정한다.

    Kaggle·Colab·JupyterHub 어디서 열려도 축 이름이 네모로 깨지지 않게 하는
    것이 목적이다. 한글 폰트를 못 찾으면 이 모듈이 만드는 라벨을 **영문으로
    바꿔** 그린다. 경고를 쏟아내는 대신 읽을 수 있는 그림을 내놓는다.

    Returns
    -------
    적용된 폰트 이름. 한글 폰트가 없으면 None.
    """
    global _HAS_KOREAN
    import warnings

    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            _HAS_KOREAN = True
            return name

    try:  # 있으면 쓴다. 없다고 해서 설치를 시도하지는 않는다.
        import koreanize_matplotlib  # noqa: F401

        _HAS_KOREAN = True
        return matplotlib.rcParams["font.family"][0]
    except ImportError:
        pass

    _HAS_KOREAN = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    if verbose:
        print("한글 폰트를 찾지 못했습니다. 그림의 축 이름을 영문으로 표시합니다.")
    return None


def _ax(ax=None, figsize=(5.2, 3.8)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def loss_curve(history, keys=("loss", "val_loss"), ax=None, title="학습 곡선"):
    """손실 곡선.

    history는 dict이거나 Keras의 History 객체이거나, PyTorch 학습 루프에서
    직접 모은 {'loss': [...], 'val_loss': [...]} 형태 — 셋 다 받는다.
    세 판이 같은 그림을 그리게 하기 위한 장치다.
    """
    hist = getattr(history, "history", history)
    ax = _ax(ax)
    for k in keys:
        if k in hist and len(hist[k]):
            ax.plot(range(1, len(hist[k]) + 1), hist[k], marker="o", ms=3, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(_L(title))
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def confusion(cm, class_names=None, ax=None, title="혼동행렬"):
    """혼동행렬을 표로 그린다. 행=실제, 열=예측."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    names = class_names or [str(i) for i in range(n)]
    ax = _ax(ax, figsize=(0.9 * n + 2.2, 0.9 * n + 1.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel(_L("예측"))
    ax.set_ylabel(_L("실제"))
    ax.set_title(_L(title))
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    return ax


def image_grid(images, labels=None, preds=None, n=16, cols=8,
               class_names=None, cmap="gray"):
    """이미지를 격자로 보여준다. 예측이 있으면 틀린 것을 빨갛게 표시한다."""
    import matplotlib.pyplot as plt

    images = np.asarray(images)
    n = min(n, len(images))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.5 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        ax.imshow(img.squeeze(), cmap=cmap if img.squeeze().ndim == 2 else None)
        if labels is None:
            continue
        true = int(labels[i])
        name = class_names[true] if class_names else str(true)
        if preds is None:
            ax.set_title(name, fontsize=8)
        else:
            p = int(preds[i])
            pname = class_names[p] if class_names else str(p)
            ok = p == true
            ax.set_title(pname if ok else f"{pname}\n({_L('정답')} {name})",
                         fontsize=8, color="black" if ok else "crimson")
    fig.tight_layout()
    return fig


def scatter2d(x, y, ax=None, class_names=("안 익음", "익음"),
              xlabel="크기", ylabel="색깔", title=None):
    """2차원 점 뿌리기. 사과 예제의 기본 그림."""
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    ax = _ax(ax)
    markers = ("o", "^", "s", "D", "v")
    for k, cls in enumerate(np.unique(y)):
        m = y == cls
        label = _L(class_names[k]) if k < len(class_names) else str(cls)
        ax.scatter(x[m, 0], x[m, 1], s=26, marker=markers[k % len(markers)],
                   alpha=0.8, label=label, edgecolors="none")
    ax.set_xlabel(_L(xlabel))
    ax.set_ylabel(_L(ylabel))
    if title:
        ax.set_title(_L(title))
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def decision_boundary(predict, x, y, ax=None, resolution: int = 200,
                      class_names=("안 익음", "익음"),
                      xlabel="크기", ylabel="색깔", title=None):
    """모델이 그은 경계를 배경색으로 칠하고 그 위에 점을 얹는다.

    predict는 (n, 2) numpy 배열을 받아 (n,) 또는 (n, C)를 돌려주는
    **아무 함수**면 된다. Keras의 model.predict도, PyTorch 모델을 감싼
    람다도 똑같이 들어간다. 세 판이 같은 그림을 그리는 이유다.
    """
    from .metrics import to_labels

    x = np.asarray(x)
    pad = 0.4
    xs = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, resolution)
    ys = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, resolution)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()].astype("float32")
    zz = to_labels(predict(grid)).reshape(gx.shape)

    ax = _ax(ax)
    ax.contourf(gx, gy, zz, alpha=0.18, levels=np.arange(zz.max() + 2) - 0.5,
                cmap="coolwarm")
    scatter2d(x, y, ax=ax, class_names=class_names,
              xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


def roc(y_true, y_score, ax=None, title="ROC 곡선"):
    """ROC 곡선과 AUC. 보고서에 그대로 넣을 수 있는 형태로 그린다."""
    from .metrics import auc as _auc
    from .metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax = _ax(ax, figsize=(4.4, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {_auc(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray", label=_L("무작위"))
    ax.set_xlabel(_L("거짓 양성 비율 (FPR)"))
    ax.set_ylabel(_L("참 양성 비율 (TPR)"))
    ax.set_title(_L(title))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return ax
