"""평가 지표 — numpy만 쓴다. 세 판의 노트북이 같은 셀을 공유한다.

프레임워크가 제공하는 지표를 쓰지 않는 이유가 있다.
Keras의 accuracy와 PyTorch에서 직접 센 accuracy가 다르게 나오면
학생은 '어느 쪽이 맞나'를 판단할 근거가 없다.
세 판이 같은 함수로 재면 그 질문 자체가 생기지 않는다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_labels",
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "roc_curve",
    "auc",
    "mae",
    "rmse",
    "r2",
    "report",
]


def to_labels(y) -> np.ndarray:
    """모델 출력을 정수 레이블로 바꾼다.

    - (n, C) 확률/로짓  → argmax
    - (n, 1) 또는 (n,) 실수, 값이 0~1 → 0.5 기준 이진화
    - 이미 정수 레이블  → 그대로
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype("int64")
    y = y.reshape(-1)
    if np.issubdtype(y.dtype, np.integer):
        return y.astype("int64")
    return (y > 0.5).astype("int64")


def accuracy(y_true, y_pred) -> float:
    """맞힌 비율."""
    yt, yp = to_labels(y_true), to_labels(y_pred)
    return float((yt == yp).mean())


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    """혼동행렬. 행이 실제(true), 열이 예측(pred)이다.

    행/열 순서를 헷갈리면 정밀도와 재현율이 뒤바뀐다.
    이 책은 **행=실제, 열=예측**으로 통일한다.
    """
    yt, yp = to_labels(y_true), to_labels(y_pred)
    if n_classes is None:
        n_classes = int(max(yt.max(), yp.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype="int64")
    np.add.at(cm, (yt, yp), 1)
    return cm


def precision_recall_f1(y_true, y_pred, positive: int = 1) -> dict[str, float]:
    """이진 분류의 정밀도·재현율·F1.

    positive 클래스를 명시적으로 받는다. '양성이 무엇인가'는
    문제마다 다르고, 이 선택이 지표를 통째로 뒤집는다.
    """
    yt, yp = to_labels(y_true), to_labels(y_pred)
    tp = int(((yp == positive) & (yt == positive)).sum())
    fp = int(((yp == positive) & (yt != positive)).sum())
    fn = int(((yp != positive) & (yt == positive)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def roc_curve(y_true, y_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC 곡선. (fpr, tpr, threshold)를 돌려준다.

    y_score는 **이진화하기 전의 점수**여야 한다. 여기에 0/1 레이블을 넣으면
    점이 세 개뿐인 곡선이 나온다 — 학생이 자주 저지르는 실수다.
    """
    yt = to_labels(y_true)
    ys = np.asarray(y_score).reshape(-1).astype("float64")
    order = np.argsort(-ys)
    ys, yt = ys[order], yt[order]

    pos = int((yt == 1).sum())
    neg = int((yt == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("한쪽 클래스만 있습니다. ROC를 그릴 수 없습니다.")

    tps = np.cumsum(yt == 1)
    fps = np.cumsum(yt == 0)
    tpr = np.concatenate([[0.0], tps / pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / neg, [1.0]])
    thr = np.concatenate([[np.inf], ys, [-np.inf]])
    return fpr, tpr, thr


def auc(y_true, y_score) -> float:
    """ROC 곡선 아래 면적. 사다리꼴 적분."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(tpr, fpr))


def mae(y_true, y_pred) -> float:
    return float(np.abs(np.asarray(y_true, dtype="float64").reshape(-1)
                        - np.asarray(y_pred, dtype="float64").reshape(-1)).mean())


def rmse(y_true, y_pred) -> float:
    d = (np.asarray(y_true, dtype="float64").reshape(-1)
         - np.asarray(y_pred, dtype="float64").reshape(-1))
    return float(np.sqrt((d ** 2).mean()))


def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype="float64").reshape(-1)
    yp = np.asarray(y_pred, dtype="float64").reshape(-1)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0


def report(y_true, y_pred, class_names: list[str] | None = None) -> str:
    """분류 결과 한 장 요약. 노트북 마지막 셀에서 print한다."""
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    names = class_names or [str(i) for i in range(n)]
    lines = [f"정확도  {accuracy(y_true, y_pred):.4f}", "", "혼동행렬 (행=실제, 열=예측)"]
    width = max(6, max(len(x) for x in names) + 1)
    lines.append(" " * width + "".join(f"{x:>{width}}" for x in names))
    for i, row in enumerate(cm):
        lines.append(f"{names[i]:<{width}}" + "".join(f"{v:>{width}}" for v in row))
    lines.append("")
    lines.append(f"{'클래스':<{width}}{'정밀도':>9}{'재현율':>9}{'F1':>9}{'개수':>7}")
    for i, name in enumerate(names):
        m = precision_recall_f1(to_labels(y_true) == i, to_labels(y_pred) == i)
        cnt = int((to_labels(y_true) == i).sum())
        lines.append(f"{name:<{width}}{m['precision']:>9.3f}{m['recall']:>9.3f}"
                     f"{m['f1']:>9.3f}{cnt:>7}")
    return "\n".join(lines)
