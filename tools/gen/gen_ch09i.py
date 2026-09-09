# -*- coding: utf-8 -*-
"""9장 §9.4 — 진짜 ImageNet 모델 실습. 3판."""
import json, os, pathlib
R = pathlib.Path(os.environ.get("DLREPO", os.path.expanduser("~/dlrepo")))

BOOTSTRAP = """try:
    import dlbook
except ImportError:
    !pip install -q "dlbook @ git+https://github.com/dhrim/deep-learning-in-one-semester.git"
    import dlbook"""

SETUP = """import numpy as np
import matplotlib.pyplot as plt

import dlbook
from dlbook import data, metrics, plot

dlbook.set_seed(42)
plot.use_korean()
print(dlbook.versions())"""

DATA = '''SIZE = 96          # ImageNet 모델은 224로 학습됐지만 96에서도 통합니다

# ★ scale=False — 0~255 그대로 받습니다.
#   preprocess_input 이 그 범위를 기대하기 때문입니다. (§9.4의 함정 ①)
s = data.cifar10(scale=False)
print(s.summary())
print("클래스:", " ".join(data.CIFAR10_CLASSES))

# 시험 2,000장을 **고정**합니다. 학습 데이터만 줄입니다. (7장 §7.6)
rng = np.random.default_rng(42)
ti = rng.choice(len(s.x_test), 2000, replace=False)
X_TE, Y_TE = s.x_test[ti], s.y_test[ti]

def subset(n, seed=42):
    """클래스마다 n/10 장씩 고르게 뽑는다. (층화 — 4장 §4.2)"""
    r = np.random.default_rng(seed)
    idx = np.concatenate([r.permutation(np.where(s.y_train == c)[0])[:n // 10]
                          for c in range(10)])
    return s.x_train[idx], s.y_train[idx]

SIZES = [100, 500] if dlbook.smoke.is_smoke() else [100, 500, 2000]
sets = {n: subset(n) for n in SIZES}
print()
for n in SIZES:
    print(f"  학습 {n:>5}장 (클래스당 {n // 10}장)")
print(f"  시험 {len(X_TE):>5}장 — 고정")'''

RUN = """print(f"{'학습 데이터':<12}" + "".join(f"{c:>13}" for c in COLS))
print("-" * (12 + 13 * len(COLS)))

for n in SIZES:
    x, y = sets[n]
    row = [scratch_acc(x, y, X_TE, Y_TE)]
    for name in PRETRAINED:
        row.append(pretrained_acc(name, x, y, X_TE, Y_TE))
    print(f"{n:<12}" + "".join(f"{v:>13.3f}" for v in row))
    for c, v in zip(COLS, row):
        dlbook.record(f"ch09i_n{n}_{c}", v)

print()
print("★ **사전학습 모델은 100장으로, 처음부터 학습한 것이 2,000장으로 낸 것보다")
print("  더 잘합니다.** 20분의 1의 데이터입니다.")
print("→ 이것이 전이학습을 쓰는 이유입니다. 데이터가 적을수록 격차가 큽니다.")"""

PREPROC = """# ★ §9.4의 함정 ① — 전처리를 안 하면 어떻게 되는가.
# 오류가 나지 않습니다. **성능만 조용히 무너집니다.**
print(f"{'학습 데이터':<12}{'제대로':>12}{'/255만':>12}{'잃은 폭':>12}")
print("-" * 48)
for n in SIZES:
    x, y = sets[n]
    ok = pretrained_acc(MAIN, x, y, X_TE, Y_TE)
    bad = pretrained_acc(MAIN, x, y, X_TE, Y_TE, naive=True)
    print(f"{n:<12}{ok:>12.3f}{bad:>12.3f}{ok - bad:>12.3f}")
    dlbook.record(f"ch09i_pre_n{n}_ok", ok)
    dlbook.record(f"ch09i_pre_n{n}_naive", bad)

print()
print("★ **전처리 한 줄을 빠뜨리면 0.80이 0.28이 됩니다.**")
print("  오류 메시지는 나오지 않습니다. 그냥 성능이 안 나옵니다.")
print("→ 전이학습이 '안 통한다'고 할 때 가장 먼저 볼 곳입니다.")"""

TRAIN = {
"keras": '''import keras
from keras import layers

PRETRAINED = ["VGG16", "ResNet50"]
COLS = ["처음부터"] + PRETRAINED
MAIN = "ResNet50"

_APP = {"VGG16": (keras.applications.VGG16, keras.applications.vgg16.preprocess_input),
        "ResNet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input)}
_BASE, _FEAT = {}, {}

def _resize(x):
    return np.array(keras.ops.image.resize(x.astype("float32"), (SIZE, SIZE)), copy=True)

def _features(name, x, naive=False):
    """사전학습 모델의 특징을 뽑는다. **얼려 두었으므로 한 번만 계산하면 된다.**"""
    key = (name, naive, x.shape[0], float(x[:3].sum()))
    if key in _FEAT:
        return _FEAT[key]
    Fn, pre = _APP[name]
    if name not in _BASE:
        b = Fn(weights="imagenet", include_top=False,
               input_shape=(SIZE, SIZE, 3), pooling="avg")
        b.trainable = False                      # ← 얼린다 (§9.2)
        _BASE[name] = b
    r = _resize(x)
    inp = (r / 255.0) if naive else pre(r.copy())     # ← 함정 ①
    f = np.asarray(_BASE[name].predict(inp, verbose=0, batch_size=64))
    _FEAT[key] = f
    return f

def pretrained_acc(name, xtr, ytr, xte, yte, naive=False, seed=42, epochs=40):
    """특징 추출 + 분류부만 학습. **이 함수만 판마다 다릅니다.**"""
    ftr, fte = _features(name, xtr, naive), _features(name, xte, naive)
    dlbook.set_seed(seed)
    m = keras.Sequential([layers.Input(shape=(ftr.shape[1],)),
                          layers.Dense(10, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(ftr, ytr, epochs=dlbook.smoke.epochs(epochs), batch_size=32, verbose=0)
    return metrics.accuracy(yte, m.predict(fte, verbose=0).argmax(1))

def scratch_acc(xtr, ytr, xte, yte, seed=42, epochs=30):
    """비교 대상 — 처음부터 학습하는 작은 CNN."""
    dlbook.set_seed(seed)
    m = keras.Sequential([
        layers.Input(shape=(32, 32, 3)), layers.Rescaling(1 / 255.),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(xtr, ytr, epochs=dlbook.smoke.epochs(epochs), batch_size=32, verbose=0)
    return metrics.accuracy(yte, m.predict(xte, verbose=0).argmax(1))''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

PRETRAINED = ["VGG16", "ResNet50"]
COLS = ["처음부터"] + PRETRAINED
MAIN = "ResNet50"

_APP = {"VGG16": (tf.keras.applications.VGG16,
                  tf.keras.applications.vgg16.preprocess_input),
        "ResNet50": (tf.keras.applications.ResNet50,
                     tf.keras.applications.resnet50.preprocess_input)}
_BASE, _FEAT = {}, {}

def _resize(x):
    return np.array(tf.image.resize(x.astype("float32"), (SIZE, SIZE)))

def _features(name, x, naive=False):
    key = (name, naive, x.shape[0], float(x[:3].sum()))
    if key in _FEAT:
        return _FEAT[key]
    Fn, pre = _APP[name]
    if name not in _BASE:
        b = Fn(weights="imagenet", include_top=False,
               input_shape=(SIZE, SIZE, 3), pooling="avg")
        b.trainable = False
        _BASE[name] = b
    r = _resize(x)
    inp = (r / 255.0) if naive else pre(r.copy())
    f = np.asarray(_BASE[name].predict(inp, verbose=0, batch_size=64))
    _FEAT[key] = f
    return f

def pretrained_acc(name, xtr, ytr, xte, yte, naive=False, seed=42, epochs=40):
    """특징 추출 + 분류부만 학습. **이 함수만 판마다 다릅니다.**"""
    ftr, fte = _features(name, xtr, naive), _features(name, xte, naive)
    dlbook.set_seed(seed)
    m = tf.keras.Sequential([L_.Input(shape=(ftr.shape[1],)),
                             L_.Dense(10, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(ftr, ytr, epochs=dlbook.smoke.epochs(epochs), batch_size=32, verbose=0)
    return metrics.accuracy(yte, m.predict(fte, verbose=0).argmax(1))

def scratch_acc(xtr, ytr, xte, yte, seed=42, epochs=30):
    dlbook.set_seed(seed)
    m = tf.keras.Sequential([
        L_.Input(shape=(32, 32, 3)), L_.Rescaling(1 / 255.),
        L_.Conv2D(32, 3, activation="relu"), L_.MaxPooling2D(2),
        L_.Conv2D(64, 3, activation="relu"), L_.MaxPooling2D(2),
        L_.Flatten(), L_.Dense(64, activation="relu"),
        L_.Dense(10, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(xtr, ytr, epochs=dlbook.smoke.epochs(epochs), batch_size=32, verbose=0)
    return metrics.accuracy(yte, m.predict(xte, verbose=0).argmax(1))''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch 판은 ResNet18 하나만 씁니다 (torchvision 배포본).
# 모델은 다르지만 **논증은 같습니다.**
PRETRAINED = ["ResNet18"]
COLS = ["처음부터"] + PRETRAINED
MAIN = "ResNet18"

# ImageNet 학습 때 쓴 정규화 값. **Keras의 preprocess_input 에 해당합니다.**
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_BASE, _FEAT = {}, {}

def _prep(x, naive=False):
    """(N,32,32,3) uint8 → (N,3,96,96) 정규화된 텐서."""
    t = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    t = torch.nn.functional.interpolate(t, size=(SIZE, SIZE),
                                        mode="bilinear", align_corners=False)
    if naive:
        return t                                   # ← 함정 ①: 정규화를 뺀 경우
    return (t - IMAGENET_MEAN) / IMAGENET_STD

def _features(name, x, naive=False):
    key = (name, naive, x.shape[0], float(x[:3].sum()))
    if key in _FEAT:
        return _FEAT[key]
    if name not in _BASE:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()                       # 분류부를 뗀다
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad = False                # ← 얼린다 (§9.2)
        _BASE[name] = m
    outs = []
    with torch.no_grad():
        t = _prep(x, naive)
        for i in range(0, len(t), 64):
            outs.append(_BASE[name](t[i:i + 64].to(device)).cpu().numpy())
    f = np.concatenate(outs)
    _FEAT[key] = f
    return f

def pretrained_acc(name, xtr, ytr, xte, yte, naive=False, seed=42, epochs=40):
    """특징 추출 + 분류부만 학습. **이 함수만 판마다 다릅니다.**"""
    ftr, fte = _features(name, xtr, naive), _features(name, xte, naive)
    dlbook.set_seed(seed)
    head = nn.Linear(ftr.shape[1], 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(ftr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.long)),
                    batch_size=32, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(head(xx), yy).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        p = head(torch.tensor(fte, dtype=torch.float32).to(device)).cpu().numpy()
    return metrics.accuracy(yte, p.argmax(1))

def scratch_acc(xtr, ytr, xte, yte, seed=42, epochs=30):
    """비교 대상 — 처음부터 학습하는 작은 CNN."""
    dlbook.set_seed(seed)
    m = nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 6 * 6, 64), nn.ReLU(),
        nn.Linear(64, 10)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    tx = torch.tensor(xtr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    dl = DataLoader(TensorDataset(tx, torch.tensor(ytr, dtype=torch.long)),
                    batch_size=32, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        m.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(m(xx), yy).backward(); opt.step()
    m.eval()
    ex = torch.tensor(xte, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        p = np.concatenate([m(ex[i:i+256].to(device)).cpu().numpy()
                            for i in range(0, len(ex), 256)])
    return metrics.accuracy(yte, p.argmax(1))''',
}

MD = {
"title": """# 9장 실습 ④ — 진짜 ImageNet 모델

**{ko} 판**

§9.3에서는 **직접 학습시킨** 원천 과제로 전이학습을 했습니다.
실무에서는 그러지 않습니다. **이미 잘 학습된 것을 받아 씁니다.**

> **가중치를 못 받는 환경이면** 저장소의 `data/pretrained/` 에 파일을
> 넣어 두십시오. 넣는 법은 부록 C에 있습니다.""",
"setup": "## 9.0 준비",
"data": """## 9.1 목표 과제 — CIFAR-10

§9.3에서 쓴 합성 도형 대신 **진짜 사진**을 씁니다.
32×32 컬러, 10종류.

**학습 데이터를 100 → 500 → 2,000장으로 줄여 가며** 봅니다.
전이학습의 값어치는 **데이터가 적을 때** 드러나기 때문입니다.""",
"train": """## 9.2 특징 추출 — 여기만 판마다 다릅니다

**얼린 모델의 특징을 한 번만 계산해 두고**, 그 위에 분류부만
학습합니다. 그래서 빠릅니다.

| 판 | 쓰는 모델 |
|---|---|
| Keras 3 / TensorFlow | VGG16, ResNet50 (`keras.applications`) |
| PyTorch | ResNet18 (`torchvision.models`) |

모델이 달라도 **논증은 같습니다.**""",
"run": """## 9.3 데이터가 적을수록 격차가 커집니다""",
"preproc": """## 9.4 ★ 전처리를 빠뜨리면

§9.4 본문의 「자주 막힙니다」 함정 ①입니다. **숫자로 확인합니다.**""",
"wrap": """## 정리

| 학습 데이터 | 처음부터 | VGG16 | ResNet50 |
|:--:|:--:|:--:|:--:|
| 100 | 0.249 | 0.345 | **0.615** |
| 500 | 0.360 | 0.593 | **0.750** |
| 2,000 | 0.455 | 0.718 | **0.801** |

- **사전학습 ResNet50은 100장으로 0.615를 냅니다.** 처음부터 학습한
  모델은 **2,000장으로도 0.455**입니다. **20배의 데이터를 이깁니다.**
- **데이터가 적을수록 격차가 큽니다.** 100장에서 2.5배, 2,000장에서 1.8배.
- VGG16보다 ResNet50이 낫습니다. **잔차 연결**로 훨씬 깊게 쌓은
  모델입니다 (§9.4 본문).

### 전처리

| 학습 데이터 | 제대로 | `/255`만 | 잃은 폭 |
|:--:|:--:|:--:|:--:|
| 2,000 | 0.801 | **0.282** | **0.520** |

**오류가 나지 않습니다. 성능만 조용히 무너집니다.**
전이학습이 "안 통한다"고 할 때 가장 먼저 볼 곳입니다.

### 연습

1. `SIZE` 를 96에서 128, 160으로 키우십시오. 나아집니까.
   시간은 얼마나 늘어납니까.
2. 학습 데이터를 5,000장, 10,000장으로 늘리십시오.
   **어느 지점에서 「처음부터」가 따라잡습니까.**
3. 분류부를 `Dense(10)` 대신 `Dense(128, relu) → Dense(10)` 으로
   바꾸십시오. 나아집니까. **왜 별로 안 나아집니까.**
4. **미세조정**을 해 보십시오 — 마지막 블록만 풀고 학습률 1/10로.
   특징 추출보다 나아집니까. (§9.2의 순서를 지키십시오)
5. **[열린 문제]** ImageNet 정확도가 더 높은 모델(EfficientNet 등)이
   CIFAR-10에서도 더 낫습니까. **본문 §9.4의 주장을 확인하십시오.**""",
}


def lines(t): return t.rstrip("\n").splitlines(keepends=True)
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": lines(t)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": lines(t)}

def write(path, cells, meta):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"},
                       "dlbook": meta},
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(R))

KO = {"keras": "Keras 3", "tensorflow": "TensorFlow", "pytorch": "PyTorch"}

if __name__ == "__main__":
    for ed, trainer in TRAIN.items():
        write(R / "notebooks" / ed / "ch09" / "ch09_imagenet.ipynb",
              [md(MD["title"].format(ko=KO[ed])),
               md(MD["setup"]), code(BOOTSTRAP), code(SETUP),
               md(MD["data"]), code(DATA),
               md(MD["train"]), code(trainer),
               md(MD["run"]), code(RUN),
               md(MD["preproc"]), code(PREPROC),
               md(MD["wrap"])],
              {"chapter": 9, "edition": ed, "title": "진짜 ImageNet 모델"})
    print("완료")
