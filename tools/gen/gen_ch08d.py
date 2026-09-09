# -*- coding: utf-8 -*-
"""8장 §8.8 — 표준 데이터셋 세 개에서 DNN 대 CNN. 3판."""
import json, os, pathlib
R = pathlib.Path(os.environ.get("DLREPO", os.path.expanduser("~/dlrepo")))

BOOT = """try:
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

DATA = '''# 세 데이터셋. 뒤로 갈수록 어렵습니다.
SETS = [("MNIST", data.mnist, (28, 28, 1)),
        ("Fashion-MNIST", data.fashion_mnist, (28, 28, 1)),
        ("CIFAR-10", data.cifar10, (32, 32, 3))]
if dlbook.smoke.is_smoke():
    SETS = SETS[:1]

for name, fn, shape in SETS:
    s = fn()
    print(f"{name:<16} 학습 {len(s.x_train):>6,}  시험 {len(s.x_test):>6,}  입력 {shape}")

print()
print("★ MNIST는 숫자가 대체로 **가운데** 있습니다.")
print("  Fashion-MNIST도 옷이 가운데 놓여 있습니다.")
print("  CIFAR-10은 **사진**입니다 — 배경이 있고, 각도가 다르고, 위치가 제각각입니다.")'''

RUN = """print(f"{'데이터셋':<16}{'DNN':>9}{'CNN':>9}{'차이':>9}"
      f"{'DNN 파라미터':>14}{'CNN 파라미터':>14}")
print("-" * 71)
for name, fn, shape in SETS:
    s = fn()
    a_dnn, p_dnn = train(s, shape, "dnn")
    a_cnn, p_cnn = train(s, shape, "cnn")
    key = name.split("-")[0].lower() if name != "Fashion-MNIST" else "fashion"
    print(f"{name:<16}{a_dnn:>9.3f}{a_cnn:>9.3f}{a_cnn - a_dnn:>9.3f}"
          f"{p_dnn:>14,}{p_cnn:>14,}")
    dlbook.record(f"ch08d_{key}_dnn", a_dnn)
    dlbook.record(f"ch08d_{key}_cnn", a_cnn)
    dlbook.record(f"ch08d_{key}_gap", a_cnn - a_dnn)

print()
print("★ **어려워질수록 CNN의 이점이 커집니다.**")
print("  MNIST 0.01 → Fashion 0.03 → CIFAR-10 0.20")
print()
print("★ 그리고 CIFAR-10에서 CNN은 **파라미터가 절반도 안 되는데** 이깁니다.")
print("  8장 §8.4의 파라미터 공유입니다. 크기가 아니라 **구조**가 이겼습니다.")"""

WRAP = """## 정리

| 데이터셋 | DNN | CNN | 차이 |
|---|:--:|:--:|:--:|
| MNIST | 0.976 | 0.986 | 0.010 |
| Fashion-MNIST | 0.871 | 0.898 | 0.027 |
| **CIFAR-10** | **0.468** | **0.671** | **0.203** |

- **MNIST로는 CNN이 왜 필요한지가 안 드러납니다.** 차이가 0.01뿐입니다.
  숫자가 대체로 가운데 있어 `Flatten` 이 이웃 관계를 버려도 손해가
  적기 때문입니다.
- **CIFAR-10에서 격차가 20배로 벌어집니다.** 진짜 사진은 위치도
  각도도 제각각입니다. §8.3에서 도형을 ±6픽셀 흔들었을 때 본 것과
  같은 이야기입니다.
- **CIFAR-10의 CNN은 파라미터가 DNN의 절반도 안 됩니다**
  (167,562 대 402,250). 이기는 이유가 **크기가 아니라 구조**임을
  보여 줍니다.

### 연습

1. CIFAR-10에서 **CNN 층을 하나 더** 쌓으십시오. 얼마나 나아집니까.
2. 데이터 증강(4장 §4.5)을 켜면 어느 쪽이 더 이득을 봅니까.
3. CIFAR-10에서 0.67은 낮은 값입니다. **0.9를 넘기려면 무엇이
   필요합니까.** (9장을 미리 보셔도 좋습니다)
4. Fashion-MNIST의 차이(0.027)가 MNIST(0.010)보다 큰 이유를
   두 데이터셋의 그림을 그려 보고 설명하십시오."""

T = {
"keras": '''import keras
from keras import layers

def train(s, shape, kind, seed=42, epochs=12):
    """DNN과 CNN을 같은 조건에서. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(seed)
    if kind == "dnn":
        m = keras.Sequential([
            layers.Input(shape=shape), layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")])
    else:
        m = keras.Sequential([
            layers.Input(shape=shape),
            layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2),
            layers.Flatten(), layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=128, verbose=0)
    pred = m.predict(s.x_test, verbose=0, batch_size=256).argmax(1)
    return metrics.accuracy(s.y_test, pred), m.count_params()''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def train(s, shape, kind, seed=42, epochs=12):
    """DNN과 CNN을 같은 조건에서. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(seed)
    if kind == "dnn":
        m = tf.keras.Sequential([
            L_.Input(shape=shape), L_.Flatten(),
            L_.Dense(128, activation="relu"), L_.Dense(64, activation="relu"),
            L_.Dense(10, activation="softmax")])
    else:
        m = tf.keras.Sequential([
            L_.Input(shape=shape),
            L_.Conv2D(32, 3, activation="relu"), L_.MaxPooling2D(2),
            L_.Conv2D(64, 3, activation="relu"), L_.MaxPooling2D(2),
            L_.Flatten(), L_.Dense(64, activation="relu"),
            L_.Dense(10, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=128, verbose=0)
    pred = m.predict(s.x_test, verbose=0, batch_size=256).argmax(1)
    return metrics.accuracy(s.y_test, pred), m.count_params()''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(s, shape, kind, seed=42, epochs=12):
    """DNN과 CNN을 같은 조건에서. **이 함수만 판마다 다릅니다.**

    dlbook.data 는 channels_last 로 돌려줍니다.
    PyTorch는 channels_first 라 **여기서 축을 바꿉니다.** (8장 §8.1)
    """
    dlbook.set_seed(seed)
    h, w, c = shape
    if kind == "dnn":
        m = nn.Sequential(nn.Flatten(), nn.Linear(h * w * c, 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    else:
        hh = ((h - 2) // 2 - 2) // 2
        ww = ((w - 2) // 2 - 2) // 2
        m = nn.Sequential(
            nn.Conv2d(c, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * hh * ww, 64), nn.ReLU(),
            nn.Linear(64, 10))
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    tx = torch.tensor(s.x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    dl = DataLoader(TensorDataset(tx, torch.tensor(s.y_train, dtype=torch.long)),
                    batch_size=128, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        m.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(m(xx), yy).backward(); opt.step()

    m.eval()
    ex = torch.tensor(s.x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    with torch.no_grad():
        pred = np.concatenate([m(ex[i:i + 256].to(device)).cpu().numpy()
                               for i in range(0, len(ex), 256)]).argmax(1)
    return metrics.accuracy(s.y_test, pred), sum(p.numel() for p in m.parameters())''',
}

MD = {
"title": """# 8장 실습 ④ — 표준 데이터셋에서 DNN 대 CNN

**{ko} 판**

§8.3에서는 **합성 도형**으로 확인했습니다. 이제 진짜 데이터로 합니다.

> **MNIST 하나만 보면 CNN이 별것 아닌 것처럼 보입니다.**
> 세 개를 나란히 놓아야 드러납니다.""",
"setup": "## 8.0 준비",
"data": "## 8.1 세 데이터셋",
"train": "## 8.2 학습 함수 — 여기만 판마다 다릅니다",
"run": "## 8.3 나란히 놓고 봅니다",
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
    for ed, tr in T.items():
        write(R / "notebooks" / ed / "ch08" / "ch08_datasets.ipynb",
              [md(MD["title"].format(ko=KO[ed])),
               md(MD["setup"]), code(BOOT), code(SETUP),
               md(MD["data"]), code(DATA),
               md(MD["train"]), code(tr),
               md(MD["run"]), code(RUN),
               md(WRAP)],
              {"chapter": 8, "edition": ed, "title": "표준 데이터셋 세 개"})
    print("완료")
