"""데이터 적재 — 세 판이 공유하는 하나의 진입점.

모든 함수는 **numpy 배열**을 돌려준다. tf.Tensor도 torch.Tensor도 아니다.
프레임워크로 넘기는 변환은 각 판의 노트북에서 한 줄로 한다.

책의 서사에 쓰이는 합성 데이터(귤·사과·XOR)를 여기에 함께 둔다.
2장에서 손으로 따라가는 그 예제를, 코드에서도 같은 숫자로 만난다.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from urllib.request import urlretrieve

import numpy as np

__all__ = [
    "Split",
    "split",
    "tangerines",
    "apples",
    "xor",
    "spirals",
    "shapes",
    "shape_names",
    "data_dirs",
    "find_local",
    "pretrained_dir",
    "use_local_weights",
    "fetch_all",
    "fetch_weights",
    "sine_series",
    "memory_task",
    "toy_reviews",
    "TOY_VOCAB",
    "toy_decode",
    "mnist",
    "fashion_mnist",
    "cifar10",
    "imdb",
]


# ---------------------------------------------------------------------------
# 분할
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """train / validation / test 세 벌.

    이 책은 validation과 test를 절대 같은 것으로 쓰지 않는다.
    (기존 노트북 240개 중 30개가 test를 validation으로 넘기고 있었다.
     그 실수는 7장에서 교보재로 다시 다룬다.)
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - 표시용
        return (
            f"Split(train={len(self.x_train)}, "
            f"val={len(self.x_val)}, test={len(self.x_test)})"
        )

    def summary(self) -> str:
        rows = [
            ("train", self.x_train, self.y_train),
            ("val", self.x_val, self.y_val),
            ("test", self.x_test, self.y_test),
        ]
        lines = [f"{'구분':<8}{'개수':>8}  입력 shape"]
        for name, x, _ in rows:
            lines.append(f"{name:<8}{len(x):>8}  {x.shape[1:]}")
        return "\n".join(lines)


def split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Split:
    """하나의 (x, y)를 train/val/test 셋으로 나눈다.

    정규화·스케일링은 **이 함수 뒤에** 하고, 반드시 train의 통계로만 한다.
    분할 전에 전체를 정규화하면 test의 정보가 train으로 새어 든다(데이터 누수).
    4장에서 이 실수를 직접 재현해 본다.
    """
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio와 test_ratio는 0 이상 1 미만이어야 합니다.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio가 1 이상입니다. 학습 데이터가 남지 않습니다.")

    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    if stratify and y.ndim == 1 and len(np.unique(y)) <= 50:
        idx_parts: list[np.ndarray] = [[], [], []]  # type: ignore[list-item]
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            idx_parts[0].append(idx[n_test + n_val :])  # type: ignore[union-attr]
            idx_parts[1].append(idx[n_test : n_test + n_val])  # type: ignore[union-attr]
            idx_parts[2].append(idx[:n_test])  # type: ignore[union-attr]
        tr, va, te = (np.concatenate(p) for p in idx_parts)
        for part in (tr, va, te):
            rng.shuffle(part)
    else:
        idx = rng.permutation(len(x))
        n = len(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        te, va, tr = idx[:n_test], idx[n_test : n_test + n_val], idx[n_test + n_val :]

    return Split(x[tr], y[tr], x[va], y[va], x[te], y[te])


# ---------------------------------------------------------------------------
# 책의 서사에 쓰이는 합성 데이터
# ---------------------------------------------------------------------------


def tangerines(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """귤 — 1차원. 크기 하나로 익은 것과 안 익은 것이 갈린다.

    2.2절. 기준은 '점'이다 (크기 7.5cm).

    Returns
    -------
    x : (n, 1) 크기(cm)
    y : (n,)   1이면 익은 것
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    unripe = rng.normal(6.0, 0.8, half)
    ripe = rng.normal(9.0, 0.8, n - half)
    x = np.concatenate([unripe, ripe]).reshape(-1, 1)
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    order = rng.permutation(n)
    return x[order].astype("float32"), y[order].astype("int64")


def apples(
    n: int = 200, seed: int = 42, noise: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """사과 — 2차원. 크기만으로는 안 갈려서 색깔을 축으로 더했다.

    2.4~2.6절. 기준이 '점'에서 '선'이 된다: x1 + x2 = 3.5

    슬라이드의 네 점 (2,2)→4, (3,1.3)→4.3, (1,1)→2, (2,0.7)→2.7 과
    같은 좌표계다. 값의 범위는 0~4.

    Returns
    -------
    x : (n, 2) [크기, 색깔]
    y : (n,)   1이면 익은 것
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 4.0, size=(n, 2))
    margin = x[:, 0] + x[:, 1] - 3.5
    flip = rng.normal(0.0, noise, size=n)
    y = ((margin + flip) > 0).astype("int64")
    return x.astype("float32"), y


def xor(n: int = 400, seed: int = 42, noise: float = 0.25):
    """XOR — 선 하나로는 절대 못 가르는 문제.

    2.13절. 은닉층과 비선형 활성화가 왜 필요한지를 보여주는 최소 예제.
    네 모서리 (0,0)·(0,1)·(1,0)·(1,1) 주변에 점을 뿌린다.
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
    labels = np.array([0, 1, 1, 0], dtype="int64")
    per = n // 4
    x = np.concatenate([c + rng.normal(0, noise, (per, 2)) for c in corners])
    y = np.concatenate([np.full(per, lb) for lb in labels])
    order = rng.permutation(len(x))
    return x[order].astype("float32"), y[order]


def spirals(n: int = 1200, seed: int = 42, noise: float = 0.06, turns: float = 2.4):
    """두 개의 소용돌이 — 직선으로도, 얕은 신경망으로도 잘 안 되는 문제.

    2.14절의 ④번 데이터이고, 5장의 실험대다.
    이 문제는 **하이퍼파라미터가 결과를 실제로 바꿉니다.** 사과 데이터는
    무엇을 해도 0.95가 나와서 학습률이나 옵티마이저의 차이가 안 보인다.

    인터넷 없이 만들 수 있다는 것도 중요하다. 실습 환경이 외부 접속을 막아
    두었더라도 이 데이터로는 5장 전체를 돌려볼 수 있다.
    """
    rng = np.random.default_rng(seed)
    per = n // 2
    out, lab = [], []
    for k in range(2):
        t = np.sqrt(rng.uniform(0.04, 1.0, per)) * turns * np.pi
        r = t / (turns * np.pi) * 4.0
        th = t + k * np.pi
        pts = np.c_[r * np.cos(th), r * np.sin(th)]
        out.append(pts + rng.normal(0, noise * (1 + r[:, None]), (per, 2)))
        lab.append(np.full(per, k))
    x = np.vstack(out).astype("float32")
    y = np.concatenate(lab).astype("int64")
    order = rng.permutation(len(x))
    return x[order], y[order]


def shapes(n: int = 6000, size: int = 28, seed: int = 42, noise: float = 0.12,
           shift: int = 6, classes=(0, 1, 2)):
    """도형 영상 — 인터넷 없이 만드는 영상 분류 데이터셋.

    28x28 회색조 영상에 도형 하나를 그린다.
    위치가 매번 다르고(shift), 크기와 회전도 조금씩 다르며, 잡음이 섞인다.

    도형은 다섯 가지가 있다 — 0 원, 1 사각형, 2 삼각형, 3 십자, 4 마름모.
    `classes` 로 어느 것을 쓸지 고른다. 9장의 전이학습 실습이 이 기능을 쓴다:
    **0~2로 미리 학습해 두고, 3~4를 적은 데이터로 배우게** 한다.

    왜 이런 것을 두는가
    -------------------
    MNIST를 못 받는 환경이 실제로 있다. 사내망, 폐쇄망, 그리고 이 책의 CI가
    그렇다. 그런 곳에서도 8장 전체를 돌려볼 수 있어야 한다.

    그리고 교육적으로도 쓸모가 있다. **위치가 매번 다르다**는 성질 때문에
    DNN(평탄화 후 Dense)은 고전하고 CNN은 잘한다. 8.3절에서 그 차이를 본다.

    Parameters
    ----------
    classes:
        쓸 도형의 번호들. 반환되는 레이블은 **0부터 다시 매겨진다.**
        `classes=(3, 4)` 이면 십자가 0, 마름모가 1이 된다.

    Returns
    -------
    x : (n, size, size, 1) float32, 0~1
    y : (n,) int64  — classes 안에서의 순번
    """
    classes = tuple(classes)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype("float32")
    imgs = np.zeros((n, size, size), dtype="float32")
    idx = rng.integers(0, len(classes), n)
    labels = np.array(classes, dtype="int64")[idx]

    for i in range(n):
        r = rng.uniform(4.0, 6.5)
        cy = size / 2 + rng.integers(-shift, shift + 1)
        cx = size / 2 + rng.integers(-shift, shift + 1)
        dy, dx = yy - cy, xx - cx
        cls = labels[i]

        if cls == 0:                                   # 원
            mask = (dy ** 2 + dx ** 2) <= r ** 2
        else:
            th = rng.uniform(0, 2 * np.pi)
            ry = dy * np.cos(th) - dx * np.sin(th)
            rx = dy * np.sin(th) + dx * np.cos(th)
            if cls == 1:                               # 사각형
                mask = (np.abs(ry) <= r * 0.85) & (np.abs(rx) <= r * 0.85)
            elif cls == 2:                             # 삼각형
                mask = (ry <= r * 0.8) & (ry >= -r * 0.8 - 1.7 * rx) \
                       & (ry >= -r * 0.8 + 1.7 * rx)
            elif cls == 3:                             # 십자
                arm = r * 0.32
                mask = ((np.abs(ry) <= arm) & (np.abs(rx) <= r)) | \
                       ((np.abs(rx) <= arm) & (np.abs(ry) <= r))
            else:                                      # 마름모
                mask = (np.abs(ry) + np.abs(rx)) <= r
        img = mask.astype("float32")
        img += rng.normal(0, noise, (size, size))
        imgs[i] = np.clip(img, 0.0, 1.0)

    return imgs[..., None], idx.astype("int64")


SHAPE_CLASSES = ("원", "사각형", "삼각형", "십자", "마름모")


def shape_names(classes=(0, 1, 2)):
    """`shapes(classes=...)` 가 돌려준 레이블에 맞는 이름 목록."""
    return [SHAPE_CLASSES[c] for c in classes]


# ---------------------------------------------------------------------------
# 순차열 데이터 (10장)
# ---------------------------------------------------------------------------


def sine_series(n: int = 2000, length: int = 40, seed: int = 42,
                noise: float = 0.08, horizon: int = 1):
    """주기가 섞인 파형에서 **다음 값**을 맞히는 회귀 문제.

    진폭·주기·위상이 표본마다 다르고 잡음이 섞인다.
    10.4절의 시계열 예측 실습이 쓴다.

    Returns
    -------
    x : (n, length, 1) float32 — 지난 length개
    y : (n,) float32           — 그다음 horizon 번째 값
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length + horizon, dtype="float32")
    xs, ys = [], []
    for _ in range(n):
        amp = rng.uniform(0.5, 1.5)
        period = rng.uniform(8, 25)
        phase = rng.uniform(0, 2 * np.pi)
        trend = rng.uniform(-0.01, 0.01)
        wave = amp * np.sin(2 * np.pi * t / period + phase) + trend * t
        wave = wave + rng.normal(0, noise, len(t))
        xs.append(wave[:length])
        ys.append(wave[length + horizon - 1])
    x = np.asarray(xs, dtype="float32")[..., None]
    return x, np.asarray(ys, dtype="float32")


def memory_task(n: int = 4000, length: int = 20, seed: int = 42,
                noise: float = 0.15, early: float = 0.25):
    """**표시된 자리의 값**을 기억해야 풀리는 문제. 10.5절의 실험대.

    순차열은 대부분 잡음이다. 그 안의 **한 자리**에만 신호(+1 또는 −1)가
    놓이고, 두 번째 채널이 그 자리를 1로 표시한다. 맞혀야 하는 것은
    그 신호의 부호다.

    두 가지가 동시에 어렵다.

    1. **자리가 매번 다르다.** 8장의 도형 위치 흔들림과 같은 문제다.
       Flatten 뒤에 Dense를 붙이면 자리마다 따로 배워야 한다.
    2. **표시된 자리가 앞쪽이다**(기본값: 앞 25% 안). 그러니 모델은
       그 값을 **끝까지 여러 걸음 동안 들고 가야** 한다.

    length를 늘리면 그 거리가 멀어진다. 시간 방향의 기울기 소실
    (5장 §5.4의 그것)을 눈으로 보는 장치다.

    Returns
    -------
    x : (n, length, 2) float32 — [값, 표시]
    y : (n,) int64             — 0 또는 1
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.zeros((n, length, 2), dtype="float32")
    x[:, :, 0] = rng.normal(0, noise, (n, length))
    pos = rng.integers(0, max(1, int(length * early)), n)
    x[np.arange(n), pos, 0] = np.where(y == 1, 1.0, -1.0)
    x[np.arange(n), pos, 1] = 1.0                 # 여기를 보라는 표시
    return x, y.astype("int64")


# ---------------------------------------------------------------------------
# 텍스트 (11장)
# ---------------------------------------------------------------------------

_POS = ["좋다", "훌륭하다", "재미있다", "감동적이다", "만족스럽다"]
_NEG = ["나쁘다", "지루하다", "실망이다", "형편없다", "아쉽다"]
_FILLER = ["영화", "배우", "연출", "이야기", "장면", "음악", "정말", "그냥",
           "조금", "매우", "어제", "친구랑", "다시", "역시", "그리고"]
_NEGATOR = "안"

TOY_VOCAB = ["<pad>", _NEGATOR] + _POS + _NEG + _FILLER


def toy_reviews(n: int = 6000, length: int = 16, seed: int = 42,
                hard: bool = False):
    """어순이 정답을 바꾸는 감성 분류 — 인터넷 없이 만드는 텍스트 데이터.

    **단어 가방(bag of words)으로는 원리적으로 풀 수 없습니다.**
    문장에 든 단어의 집합이 같은데 정답이 다른 짝이 있기 때문이다.
    **순서를 봐야 한다.** 11장에서 임베딩과 순차 모델이 왜 필요한지를
    보이는 장치다.

    hard=False (기본) — 「부정어가 무엇에 붙었는가」
        감성 단어 하나와 미끼 단어 하나가 있고, 부정어 "안"이 **둘 중
        하나 앞에** 붙는다. 정답은 감성 단어의 부호이며, 그 단어가
        부정되었으면 뒤집힌다.

            영화 안 좋다 정말 배우   → 부정
            영화 좋다 안 정말 배우   → 긍정      (단어 집합이 같다)

    hard=True — 「먼저 나온 것이 무엇인가」
        감성 단어가 **둘** 들어가고(부호가 서로 반대), 각각 부정될 수
        있다. 정답은 **먼저 나온** 감성 단어를 따른다.
        순서를 처음부터 끝까지 유지해야 풀리므로 훨씬 어렵고,
        **학습이 자주 실패한다**(10장 §10.5의 그 불안정성).

    Returns
    -------
    x : (n, length) int64 — 단어 번호. 0은 채움(<pad>)
    y : (n,) int64        — 0 부정, 1 긍정
    """
    rng = np.random.default_rng(seed)
    idx = {w: i for i, w in enumerate(TOY_VOCAB)}
    xs = np.zeros((n, length), dtype="int64")
    ys = np.zeros(n, dtype="int64")

    for i in range(n):
        if hard:
            first_pos = bool(rng.integers(0, 2))      # 첫 감성어가 긍정인가
            neg_first = bool(rng.integers(0, 2))      # 첫 감성어가 부정되는가
            neg_second = bool(rng.integers(0, 2))
            w1 = rng.choice(_POS if first_pos else _NEG)
            w2 = rng.choice(_NEG if first_pos else _POS)

            toks = list(rng.choice(_FILLER, rng.integers(1, 4)))
            if neg_first:
                toks.append(_NEGATOR)
            toks.append(w1)
            toks += list(rng.choice(_FILLER, rng.integers(1, 4)))
            if neg_second:
                toks.append(_NEGATOR)
            toks.append(w2)
            toks += list(rng.choice(_FILLER, rng.integers(0, 3)))
            label = int(first_pos != neg_first)
        else:
            positive = bool(rng.integers(0, 2))       # 감성어가 긍정인가
            on_sent = bool(rng.integers(0, 2))        # "안"이 감성어에 붙는가
            sent = rng.choice(_POS if positive else _NEG)
            decoy = rng.choice(_FILLER)

            toks = list(rng.choice(_FILLER, rng.integers(1, 4)))
            if on_sent:                               # ... 안 좋다 ... 배우 ...
                toks += [_NEGATOR, sent]
                toks += list(rng.choice(_FILLER, rng.integers(1, 4)))
                toks += [decoy]
            else:                                     # ... 좋다 ... 안 배우 ...
                toks += [sent]
                toks += list(rng.choice(_FILLER, rng.integers(1, 4)))
                toks += [_NEGATOR, decoy]
            toks += list(rng.choice(_FILLER, rng.integers(0, 3)))
            label = int(positive != on_sent)          # 부정되면 뒤집힌다

        toks = toks[:length]
        xs[i, :len(toks)] = [idx[t] for t in toks]
        ys[i] = label

    return xs, ys


def toy_decode(seq):
    """번호 열을 단어로 되돌린다. 채움(0)은 버린다."""
    return " ".join(TOY_VOCAB[i] for i in seq if i != 0)


# ---------------------------------------------------------------------------
# 표준 데이터셋 — keras.datasets를 통해 받아 numpy로 돌려준다
# ---------------------------------------------------------------------------


_CACHE = pathlib.Path.home() / ".dlbook" / "datasets"
_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def data_dirs() -> list[pathlib.Path]:
    """데이터 파일을 찾을 곳들. 앞에 있는 것부터 뒤진다.

    1. 환경변수 DLBOOK_DATA 가 가리키는 곳
    2. 저장소의 data/ 폴더
    3. 현재 폴더의 data/
    4. ~/.dlbook/datasets (내려받은 것을 두는 캐시)

    **인터넷이 막힌 곳에서는 파일을 2번 자리에 넣어 두면 됩니다.**
    사내망·폐쇄망에서 실습하실 때 이 경로를 쓰십시오.
    """
    here = pathlib.Path(__file__).resolve().parent
    dirs = []
    env = os.environ.get("DLBOOK_DATA")
    if env:
        dirs.append(pathlib.Path(env))
    dirs += [here.parent / "data", pathlib.Path.cwd() / "data", _CACHE]
    seen, uniq = set(), []
    for d in dirs:                      # 저장소 안에서 실행하면 2·3번이 같아진다
        key = str(d)
        if key not in seen:
            seen.add(key); uniq.append(d)
    return uniq


def find_local(name: str) -> pathlib.Path | None:
    """data_dirs() 를 훑어 파일을 찾는다. 없으면 None.

    `name` 에는 하위 경로를 써도 된다 (예: "fashion-mnist/train-images-idx3-ubyte.gz").
    그 경로로 못 찾으면 **파일 이름만으로 한 번 더** 찾는다. 폐쇄망에서 파일을
    폴더 없이 통째로 부어 넣는 경우가 흔하기 때문이다.
    """
    flat = pathlib.PurePath(name).name
    for d in data_dirs():
        for cand in (d / name, d / flat):
            if cand.exists():
                return cand
    return None


def _download(url: str, name: str, quiet: bool = False) -> pathlib.Path:
    """파일을 찾는다. 없으면 내려받아 캐시에 둔다.

    **먼저 로컬을 뒤집니다.** 인터넷이 막힌 환경에서 파일을 직접 넣어 두면
    그대로 씁니다. (data_dirs() 참조)

    받는 도중에 끊겨도 **반쯤 받은 파일이 남지 않습니다.** `.part` 로 받아서
    다 받은 뒤에 제자리로 옮깁니다. 예전에는 잘린 파일이 캐시에 남아, 다음
    실행에서 그 파일을 '있는 것'으로 보고 계속 실패했습니다.
    """
    local = find_local(name)
    if local is not None:
        return local
    path = _CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if not quiet:
        print(f"내려받는 중 … {name}  ({url})")
    try:
        urlretrieve(url, part)
        part.replace(path)
    except BaseException as exc:
        part.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"'{name}' 을 찾지 못했고 내려받지도 못했습니다.\n"
            f"  받을 곳: {url}\n"
            f"  둘 곳:   {_CACHE / name}\n"
            f"  미리 받아 두려면: python -m dlbook.fetch\n"
            f"  (원인: {type(exc).__name__})"
        ) from exc
    return path


# ---------------------------------------------------------------------------
# 사전학습 가중치 — 케라스와 파이토치가 찾을 곳을 한 군데로 모은다
# ---------------------------------------------------------------------------


def pretrained_dir() -> pathlib.Path:
    """ImageNet 사전학습 가중치를 두는 곳.

    data_dirs() 안에 `pretrained/` 가 이미 있으면 그것을 쓰고,
    없으면 `~/.dlbook/pretrained` 를 쓴다.
    """
    for d in data_dirs():
        p = d / "pretrained"
        if p.is_dir():
            return p
    return _CACHE.parent / "pretrained"


def use_local_weights(path: str | os.PathLike | None = None) -> pathlib.Path:
    """케라스와 파이토치가 가중치를 찾을 곳을 이 폴더로 맞춘다.

    케라스는 `$KERAS_HOME/models/`, 파이토치는 `$TORCH_HOME/hub/checkpoints/`
    를 봅니다. 두 환경변수를 같은 폴더로 맞춰 두면 **가중치가 한 군데에 모이고,
    그 폴더를 통째로 복사해 가면 인터넷 없이도 전이학습 실습이 돕니다.**

    `import keras` / `import torch` **보다 먼저** 불러야 합니다.
    dlbook 을 먼저 import 하면 자동으로 처리됩니다(아래 참조).
    """
    p = pathlib.Path(path) if path is not None else pretrained_dir()
    (p / "models").mkdir(parents=True, exist_ok=True)
    (p / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(p)
    os.environ["TORCH_HOME"] = str(p)
    return p


def fetch_weights(quiet: bool = False) -> pathlib.Path:
    """ImageNet 사전학습 가중치를 미리 받아 둔다.

    URL 을 직접 적지 않고 **케라스와 파이토치가 스스로 받게 합니다.**
    그래야 라이브러리 판이 올라가도 주소가 어긋나지 않습니다.
    설치돼 있지 않은 쪽은 조용히 건너뜁니다.
    """
    p = use_local_weights()
    try:
        import keras  # noqa: PLC0415
        for fn in ("VGG16", "ResNet50", "MobileNetV2"):
            if not quiet:
                print(f"내려받는 중 … {fn} (ImageNet)")
            getattr(keras.applications, fn)(weights="imagenet", include_top=False)
    except ImportError:
        if not quiet:
            print("건너뜀 … 케라스가 없습니다")
    try:
        import torchvision  # noqa: PLC0415
        if not quiet:
            print("내려받는 중 … resnet18 (ImageNet)")
        torchvision.models.resnet18(weights="IMAGENET1K_V1")
    except ImportError:
        if not quiet:
            print("건너뜀 … torchvision 이 없습니다")
    return p


# ---------------------------------------------------------------------------
# 미리 받아 두기 — 강의 전날 한 번 돌리는 것
# ---------------------------------------------------------------------------


def _needed_files() -> list[tuple[str, str]]:
    """(주소, 둘 이름) 목록. 실습에 쓰이는 표준 데이터셋 전부."""
    files = [(_MNIST_URL, "mnist.npz"), (_CIFAR_URL, "cifar-10-python.tar.gz")]
    files += [(_FASHION_BASE + v, f"fashion-mnist/{v}")
              for v in _FASHION_FILES.values()]
    return files


def fetch_all(dest: str | os.PathLike | None = None,
              weights: bool = False,
              quiet: bool = False) -> pathlib.Path:
    """실습에 필요한 데이터셋을 한 번에 내려받는다.

    **강의 전날 한 번 돌려 두면, 강의실에서 인터넷이 막혀도 실습이 돕니다.**
    받은 폴더를 USB 로 옮겨 `DLBOOK_DATA` 가 가리키게 해도 됩니다.

        python -m dlbook.fetch                 # ~/.dlbook/datasets 에
        python -m dlbook.fetch --dest data/    # 저장소의 data/ 에
        python -m dlbook.fetch --weights       # 사전학습 가중치까지

    이미 있는 파일은 건너뜁니다.
    """
    d = pathlib.Path(dest) if dest is not None else _CACHE
    d.mkdir(parents=True, exist_ok=True)
    for url, name in _needed_files():
        path = d / name
        if path.exists():
            if not quiet:
                print(f"이미 있음 … {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        if not quiet:
            print(f"내려받는 중 … {name}")
        try:
            urlretrieve(url, part)
            part.replace(path)
        except BaseException as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"'{name}' 을 받지 못했습니다: {url}") from exc
    if weights:
        fetch_weights(quiet=quiet)
    if not quiet:
        print(f"\n다 받았습니다 → {d}")
        if dest is not None:
            print("이 폴더를 쓰게 하려면:  export DLBOOK_DATA=" + str(d.resolve()))
    return d


def _keras_dataset(name: str):
    """keras.datasets 로 넘어간다.

    **PyTorch만 쓰는 노트북에서 이 함수를 부르면 TensorFlow가 통째로 메모리에
    올라갑니다.** 브라우저 실습 환경에서는 그것만으로 커널이 죽습니다.
    그래서 mnist()는 keras를 거치지 않고 직접 내려받습니다(아래).
    나머지 데이터셋은 아직 keras를 거칩니다.
    """
    try:
        import keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"'{name}' 데이터셋을 받으려면 keras가 필요합니다. "
            "pip install keras 로 설치하십시오."
        ) from exc
    return getattr(keras.datasets, name)


def _to_split(
    x_train, y_train, x_test, y_test, val_ratio: float, seed: int
) -> Split:
    """표준 데이터셋은 train/test 두 벌로만 온다. train을 다시 갈라 val을 만든다."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    n_val = int(round(len(idx) * val_ratio))
    va, tr = idx[:n_val], idx[n_val:]
    return Split(
        np.asarray(x_train)[tr],
        np.asarray(y_train)[tr],
        np.asarray(x_train)[va],
        np.asarray(y_train)[va],
        np.asarray(x_test),
        np.asarray(y_test),
    )


def mnist(val_ratio: float = 0.1, seed: int = 42, scale: bool = True,
          channels: bool = True) -> Split:
    """손글씨 숫자 0~9. 28x28 회색조.

    **keras를 거치지 않고 직접 내려받습니다.** PyTorch만 쓰는 노트북에서
    TensorFlow가 딸려 오지 않게 하기 위해서입니다.
    받은 파일은 ~/.dlbook/datasets 에 캐시됩니다.
    """
    with np.load(_download(_MNIST_URL, "mnist.npz")) as f:
        xt, yt = f["x_train"], f["y_train"]
        xs, ys = f["x_test"], f["y_test"]
    if scale:
        xt = xt.astype("float32") / 255.0
        xs = xs.astype("float32") / 255.0
    if channels:                       # (N, 28, 28) → (N, 28, 28, 1)
        xt, xs = xt[..., None], xs[..., None]
    return _to_split(xt, yt.reshape(-1), xs, ys.reshape(-1), val_ratio, seed)


def _read_idx(path: pathlib.Path) -> np.ndarray:
    """IDX 형식(.gz)을 numpy 배열로. MNIST 계열의 원본 형식이다."""
    import gzip
    import struct
    with gzip.open(path, "rb") as f:
        raw = f.read()
    _, _, dtype_code, n_dims = raw[0], raw[1], raw[2], raw[3]
    if dtype_code != 0x08:                     # unsigned byte 만 쓴다
        raise ValueError(f"지원하지 않는 IDX 자료형: {dtype_code:#x}")
    dims = struct.unpack(">" + "I" * n_dims, raw[4:4 + 4 * n_dims])
    return np.frombuffer(raw, dtype=np.uint8,
                         offset=4 + 4 * n_dims).reshape(dims)


_FASHION_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
_FASHION_FILES = {
    "xt": "train-images-idx3-ubyte.gz", "yt": "train-labels-idx1-ubyte.gz",
    "xs": "t10k-images-idx3-ubyte.gz",  "ys": "t10k-labels-idx1-ubyte.gz",
}


def fashion_mnist(val_ratio: float = 0.1, seed: int = 42, scale: bool = True,
                  channels: bool = True) -> Split:
    """의류 10종. 28x28 회색조.

    **keras를 거치지 않습니다.** 원본 IDX 파일 4개를 직접 읽습니다.
    (mnist() 와 같은 이유 — PyTorch 노트북에 TensorFlow가 딸려 오면 안 됩니다.)
    """
    got = {k: _read_idx(_download(_FASHION_BASE + v, f"fashion-mnist/{v}"))
           for k, v in _FASHION_FILES.items()}
    (xt, yt), (xs, ys) = (got["xt"], got["yt"]), (got["xs"], got["ys"])
    if scale:
        xt = xt.astype("float32") / 255.0
        xs = xs.astype("float32") / 255.0
    if channels:
        xt, xs = xt[..., None], xs[..., None]
    return _to_split(xt, yt.reshape(-1), xs, ys.reshape(-1), val_ratio, seed)


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def cifar10(val_ratio: float = 0.1, seed: int = 42, scale: bool = True) -> Split:
    """사물 10종. 32x32 컬러.

    **keras를 거치지 않습니다.** 배포본 tar.gz 을 직접 풀어 읽습니다.
    """
    import pickle
    import tarfile

    path = _download(_CIFAR_URL, "cifar-10-python.tar.gz")
    xs_, ys_, xte, yte = [], [], None, None
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            name = m.name.rsplit("/", 1)[-1]
            if not (name.startswith("data_batch") or name == "test_batch"):
                continue
            d = pickle.load(tar.extractfile(m), encoding="latin1")
            # (N, 3072) → (N, 3, 32, 32) → (N, 32, 32, 3)
            arr = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            lab = np.asarray(d["labels"], dtype="int64")
            if name == "test_batch":
                xte, yte = arr, lab
            else:
                xs_.append(arr); ys_.append(lab)
    xt = np.concatenate(xs_); yt = np.concatenate(ys_)
    xs, ys = xte, yte
    if scale:
        xt = xt.astype("float32") / 255.0
        xs = xs.astype("float32") / 255.0
    return _to_split(xt, yt.reshape(-1), xs, ys.reshape(-1), val_ratio, seed)


def imdb(num_words: int = 10000, val_ratio: float = 0.1, seed: int = 42) -> Split:
    """영화평 긍/부정. 정수 인덱스 열."""
    (xt, yt), (xs, ys) = _keras_dataset("imdb").load_data(num_words=num_words)
    return _to_split(xt, np.asarray(yt), xs, np.asarray(ys), val_ratio, seed)


CIFAR10_CLASSES = (
    "비행기", "자동차", "새", "고양이", "사슴",
    "개", "개구리", "말", "배", "트럭",
)
FASHION_CLASSES = (
    "티셔츠", "바지", "풀오버", "드레스", "코트",
    "샌들", "셔츠", "운동화", "가방", "앵클부츠",
)


# ---------------------------------------------------------------------------
# import 시점의 단 한 가지 부수효과
# ---------------------------------------------------------------------------
# data_dirs() 안에 이미 `pretrained/` 가 놓여 있으면, 케라스와 파이토치가
# 그곳을 보도록 환경변수를 맞춰 둔다. 폐쇄망 강의실에서 가중치 폴더만 복사해
# 두면 전이학습 실습이 그대로 도는 것은 이 세 줄 덕분이다.
# 폴더가 없으면 아무것도 하지 않는다 — 각 라이브러리의 기본 경로를 쓴다.
if any((_d / "pretrained").is_dir() for _d in data_dirs()):
    use_local_weights()
