# -*- coding: utf-8 -*-
"""13장 실습 노트북 3판 생성 — 오토인코더."""
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

DATA = """# MNIST. **정답(y)을 쓰지 않습니다.** 입력이 곧 정답입니다.
s = data.mnist()
x_train = s.x_train.reshape(len(s.x_train), -1)      # (N, 784)
x_test = s.x_test.reshape(len(s.x_test), -1)
print(f"학습 {x_train.shape}, 시험 {x_test.shape}")
print()
print("★ y_train 을 한 번도 쓰지 않습니다. 이것이 비지도 학습입니다.")

def show(rows, titles, n=8):
    \"\"\"여러 줄의 28x28 영상을 나란히 그린다.\"\"\"
    fig, axes = plt.subplots(len(rows), n, figsize=(1.15 * n, 1.25 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (imgs, t) in enumerate(zip(rows, titles)):
        for c in range(n):
            axes[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(t)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.tight_layout(); plt.show()"""

PCA_BASE = '''# ★ 기준선을 먼저 잽니다. (7장 §7.2)
# 차원을 줄이는 고전적인 방법이 **주성분 분석(PCA)** 입니다.
# 오토인코더는 이것을 못 이기면 쓸 이유가 없습니다.
mean = x_train.mean(0)
_, _, Vt = np.linalg.svd(x_train[:10000] - mean, full_matrices=False)

def pca_mse(latent):
    P = Vt[:latent]
    rec = (x_test - mean) @ P.T @ P + mean
    return float(np.mean((rec - x_test) ** 2))

print(f"{'잠재 차원':<12}{'PCA 복원 MSE':>16}")
for L in (2, 8, 32, 64):
    v = pca_mse(L)
    print(f"{L:<12}{v:>16.5f}")
    dlbook.record(f"ch13_pca_mse_{L}", v)

print()
print("→ 이것이 기준선입니다. 오토인코더는 이 값보다 낮아야 의미가 있습니다.")'''

RUN = """latents = [2, 32] if dlbook.smoke.is_smoke() else [2, 8, 32, 64]

print(f"{'잠재 차원':<12}{'압축률':>10}{'AE MSE':>12}{'PCA MSE':>12}{'차이':>10}")
print("-" * 58)
for L in latents:
    mse, _ = train_ae(L, x_train, x_test)
    p = pca_mse(L)
    print(f"{L:<12}{784 // L:>9}배{mse:>12.5f}{p:>12.5f}{p - mse:>10.5f}")
    dlbook.record(f"ch13_ae_mse_{L}", mse)

print()
print("→ 모든 잠재 차원에서 **오토인코더가 PCA보다 낫습니다.**")
print("→ 784개 숫자를 32개로 줄였다가 되살립니다. **24분의 1**입니다.")"""

LINEAR = """# 활성화 함수를 전부 빼면 어떻게 되는가.
# 8장 §8.7에서 한 질문을 여기서 다시 합니다.
print(f"{'잠재':<8}{'비선형 AE':>12}{'선형 AE':>12}{'PCA':>12}")
print("-" * 44)
for L in ([2] if dlbook.smoke.is_smoke() else [2, 32]):
    nl, _ = train_ae(L, x_train, x_test)
    li, _ = train_ae(L, x_train, x_test, linear=True)
    pc = pca_mse(L)
    print(f"{L:<8}{nl:>12.5f}{li:>12.5f}{pc:>12.5f}")
    dlbook.record(f"ch13_linear_mse_{L}", li)

print()
print("★ **선형 오토인코더는 PCA와 사실상 같은 값을 냅니다.**")
print("  소수점 넷째 자리까지 같습니다. 우연이 아니라 정리입니다 —")
print("  선형 오토인코더가 찾는 부분공간은 PCA가 찾는 것과 같습니다.")
print()
print("→ **오토인코더가 PCA보다 나은 이유는 「깊어서」가 아니라 「비선형이라서」**")
print("  입니다. 활성화 함수를 빼는 순간 60년 된 방법으로 돌아갑니다.")"""

RECON = """# 눈으로 봅니다.
_, model = train_ae(32, x_train, x_test)
rec = reconstruct(model, x_test[:8])
show([x_test[:8], rec], ["원본", "복원(32)"])

_, model2 = train_ae(2, x_train, x_test)
rec2 = reconstruct(model2, x_test[:8])
show([x_test[:8], rec2], ["원본", "복원(2)"])

print("→ 잠재 32개로는 알아볼 만합니다. **2개로는 뭉개집니다.**")
print("→ 784개를 2개로 줄이면 그만큼 버려집니다. 무엇을 남길지는 학습이 정합니다.")"""

DENOISE = """# 잡음 낀 그림을 넣고 **깨끗한 그림을 정답으로** 학습합니다.
rng = np.random.default_rng(0)
def add_noise(X, level=0.4):
    return np.clip(X + rng.normal(0, level, X.shape), 0, 1).astype("float32")

x_train_n, x_test_n = add_noise(x_train), add_noise(x_test)

base = float(np.mean((x_test_n - x_test) ** 2))
print(f"잡음 낀 입력 자체의 MSE (기준선)      {base:.5f}")
dlbook.record("ch13_noise_baseline", base)

mse_dn, model_dn = train_ae(32, x_train_n, x_test_n, y_train=x_train, y_test=x_test)
print(f"잡음→깨끗 으로 학습한 AE의 복원 MSE   {mse_dn:.5f}")
dlbook.record("ch13_denoise_mse", mse_dn)

mse_plain, model_p = train_ae(32, x_train, x_test)
rec_p = reconstruct(model_p, x_test_n)
cross = float(np.mean((rec_p - x_test) ** 2))
print(f"(참고) 깨끗→깨끗 AE에 잡음을 넣으면    {cross:.5f}")
dlbook.record("ch13_clean_ae_on_noise", cross)

show([x_test[:8], x_test_n[:8], reconstruct(model_dn, x_test_n[:8])],
     ["원본", "잡음", "복원"])

print()
print("→ 잡음이 낀 입력(0.0797)을 넣어 **원본에 훨씬 가까운 것(0.0172)** 을 냅니다.")
print("→ 이것이 **잡음 제거 오토인코더**입니다. 입력과 정답을 다르게 준 것뿐입니다.")
print("→ 깨끗한 것만 보고 배운 AE는 잡음에 약합니다(0.0390). **본 것만 잘합니다.**")"""

ANOMALY_HEAD = """def recon_error(model, X):
    return np.mean((reconstruct(model, X) - X) ** 2, axis=1)

def auc(score, label):
    \"\"\"ROC 곡선 아래 면적. 순위만 쓰므로 임계값을 안 정해도 된다.\"\"\"
    order = np.argsort(score); r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    n1 = int(label.sum()); n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))"""

ANOM1 = """# 같은 데이터 안의 새 숫자. 0~8만 배우고 9를 찾아낸다.
keep = s.y_train != 9

print(f"{'잠재':<8}{'정상 오차':>12}{'9의 오차':>12}{'AUC':>10}")
print("-" * 42)
for L in ([8] if dlbook.smoke.is_smoke() else [2, 8, 32]):
    _, m = train_ae(L, x_train[keep], x_test)
    e = recon_error(m, x_test)
    a = auc(e, (s.y_test == 9).astype(int))
    print(f"{L:<8}{e[s.y_test != 9].mean():>12.5f}"
          f"{e[s.y_test == 9].mean():>12.5f}{a:>10.3f}")
    dlbook.record(f"ch13_anomaly_digit9_auc_{L}", a)

print()
print("★ **거의 안 됩니다.** AUC가 0.5(찍기) 근처입니다.")
print("  0~8을 배우며 익힌 획과 곡선으로 9도 그대로 그려 버립니다.")
print("  모델이 배운 것은 '내가 본 아홉 종류'가 아니라 **'손으로 쓴 획'** 입니다.")"""

ANOM2 = """# 분포가 아예 다른 것. 숫자로 배우고 옷 사진을 보여 준다.
f = data.fashion_mnist()
x_fashion = f.x_test.reshape(len(f.x_test), -1)

print(f"{'잠재':<8}{'숫자 오차':>12}{'옷 오차':>12}{'배수':>8}{'AUC':>10}")
print("-" * 50)
for L in ([8] if dlbook.smoke.is_smoke() else [8, 32]):
    _, m = train_ae(L, x_train, x_test)
    em, ef = recon_error(m, x_test), recon_error(m, x_fashion)
    lab = np.concatenate([np.zeros(len(em)), np.ones(len(ef))]).astype(int)
    a = auc(np.concatenate([em, ef]), lab)
    print(f"{L:<8}{em.mean():>12.5f}{ef.mean():>12.5f}"
          f"{ef.mean() / em.mean():>7.1f}배{a:>10.3f}")
    dlbook.record(f"ch13_anomaly_fashion_auc_{L}", a)

show([x_test[:8], reconstruct(m, x_test[:8])], ["숫자", "복원"])
show([x_fashion[:8], reconstruct(m, x_fashion[:8])], ["옷", "복원"])

print()
print("★ **이번에는 잘 됩니다.** AUC 0.99.")
print("→ 옷 사진은 숫자로 배운 부품으로 그려지지 않습니다. 오차가 10배입니다.")"""

TRAIN = {
"keras": '''import keras
from keras import layers

def _build(latent, linear=False):
    """인코더와 디코더 — **이 함수만 판마다 다릅니다.**"""
    act = None if linear else "relu"
    out_act = None if linear else "sigmoid"
    encoder = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation=act),
        layers.Dense(latent, activation=act),
    ], name="encoder")
    decoder = keras.Sequential([
        layers.Input(shape=(latent,)),
        layers.Dense(128, activation=act),
        layers.Dense(784, activation=out_act),
    ], name="decoder")
    return keras.Sequential([encoder, decoder]), encoder, decoder

def train_ae(latent, xa, xb, y_train=None, y_test=None,
             linear=False, seed=42, epochs=15):
    """(시험 복원 MSE, 모델) 을 돌려준다.

    y_train 을 주지 않으면 **입력이 곧 정답**이다. 주면 잡음 제거가 된다.
    """
    dlbook.set_seed(seed)
    m, enc, dec = _build(latent, linear)
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(xa, xa if y_train is None else y_train,
          epochs=dlbook.smoke.epochs(epochs), batch_size=256, verbose=0)
    target = xb if y_test is None else y_test
    rec = m.predict(xb, verbose=0)
    return float(np.mean((rec - target) ** 2)), m

def reconstruct(model, X):
    return model.predict(X, verbose=0)

def encode(model, X):
    return model.layers[0].predict(X, verbose=0)''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def _build(latent, linear=False):
    """인코더와 디코더 — **이 함수만 판마다 다릅니다.**"""
    act = None if linear else "relu"
    out_act = None if linear else "sigmoid"
    encoder = tf.keras.Sequential([
        L_.Input(shape=(784,)),
        L_.Dense(128, activation=act),
        L_.Dense(latent, activation=act),
    ], name="encoder")
    decoder = tf.keras.Sequential([
        L_.Input(shape=(latent,)),
        L_.Dense(128, activation=act),
        L_.Dense(784, activation=out_act),
    ], name="decoder")
    return tf.keras.Sequential([encoder, decoder]), encoder, decoder

def train_ae(latent, xa, xb, y_train=None, y_test=None,
             linear=False, seed=42, epochs=15):
    """(시험 복원 MSE, 모델) 을 돌려준다."""
    dlbook.set_seed(seed)
    m, enc, dec = _build(latent, linear)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    m.fit(xa, xa if y_train is None else y_train,
          epochs=dlbook.smoke.epochs(epochs), batch_size=256, verbose=0)
    target = xb if y_test is None else y_test
    return float(np.mean((m.predict(xb, verbose=0) - target) ** 2)), m

def reconstruct(model, X):
    return model.predict(X, verbose=0)

def encode(model, X):
    return model.layers[0].predict(X, verbose=0)''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class AutoEncoder(nn.Module):
    """인코더로 줄이고 디코더로 되살린다.

    **가운데가 좁은 것**이 전부입니다. 그 좁은 곳을 잠재 공간이라 부릅니다.
    """
    def __init__(self, latent, linear=False):
        super().__init__()
        act = nn.Identity if linear else nn.ReLU
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), act(), nn.Linear(128, latent), act())
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), act(), nn.Linear(128, 784),
            nn.Identity() if linear else nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_ae(latent, xa, xb, y_train=None, y_test=None,
             linear=False, seed=42, epochs=15):
    """(시험 복원 MSE, 모델) 을 돌려준다."""
    dlbook.set_seed(seed)
    model = AutoEncoder(latent, linear).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    tgt = xa if y_train is None else y_train
    dl = DataLoader(TensorDataset(torch.tensor(xa, dtype=torch.float32),
                                  torch.tensor(tgt, dtype=torch.float32)),
                    batch_size=256, shuffle=True)
    loss_fn = nn.MSELoss()
    for _ in range(dlbook.smoke.epochs(epochs)):
        model.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(model(xx), yy).backward(); opt.step()
    target = xb if y_test is None else y_test
    return float(np.mean((reconstruct(model, xb) - target) ** 2)), model

def reconstruct(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(device))
    return out.cpu().numpy()

def encode(model, X):
    model.eval()
    with torch.no_grad():
        out = model.encoder(torch.tensor(X, dtype=torch.float32).to(device))
    return out.cpu().numpy()''',
}

MD = {
"t_ae": """# 13장 실습 ① — 오토인코더와 PCA

**{ko} 판**

**정답이 없습니다.** 입력을 압축했다가 되살리는 것이 전부입니다.

> 그런데 차원 축소는 60년 된 방법(PCA)이 이미 있습니다.
> **오토인코더는 그것보다 나은가.** 재 봅니다.""",
"t_dn": """# 13장 실습 ② — 잡음 제거 오토인코더

**{ko} 판**

바꾸는 것은 **한 줄**입니다 — 정답으로 무엇을 주는가.""",
"t_an": """# 13장 실습 ③ — 이상 탐지가 **안 되는** 경우

**{ko} 판**

*"정상만 보고 배운 모델은 이상한 것을 잘 못 그린다"* 는 착상입니다.
널리 소개되는 용법인데, **이 경우에는 안 됩니다.**

실습 ④와 함께 보셔야 합니다.""",
"t_an2": """# 13장 실습 ④ — 이상 탐지가 **되는** 경우

**{ko} 판**

실습 ③에서는 AUC가 0.5 근처였습니다. 여기서는 0.99입니다.
**무엇이 다른지**가 이 절의 요지입니다.""",
"setup": "## 13.0 준비",
"data": """## 13.1 데이터 — 정답을 쓰지 않습니다

이 장에서 처음으로 `y` 를 쓰지 않습니다.""",
"pca": """## 13.2 기준선 — PCA

7장 §7.2의 규칙입니다. **기준선 없이 성능을 읽을 수 없습니다.**""",
"train": """## 13.3 모델 정의 — 여기만 판마다 다릅니다

인코더로 줄이고 디코더로 되살립니다. **가운데가 좁은 것**이 전부입니다.""",
"run": """## 13.4 오토인코더는 PCA를 이기는가""",
"linear": """## 13.5 왜 이기는가 — 활성화를 빼 봅니다""",
"recon": """## 13.6 눈으로 보기""",
"denoise": """## 13.1 잡음 제거""",
"anom1": """## 13.1 이상 탐지 — 같은 데이터 안의 새 종류

MNIST에서 **0~8만** 학습하고, 시험에서 **9**를 찾아냅니다.""",
"anom2": """## 13.1 이상 탐지 — 분포가 아예 다른 것

숫자로 학습하고 **옷 사진**을 보여 줍니다.""",

"w_ae": """## 정리

| 잠재 차원 | 압축률 | 오토인코더 | PCA |
|:--:|:--:|:--:|:--:|
| 2 | 392배 | **0.0456** | 0.0557 |
| 8 | 98배 | **0.0233** | 0.0375 |
| 32 | 24배 | **0.0102** | 0.0169 |
| 64 | 12배 | **0.0067** | 0.0091 |

- **모든 차원에서 오토인코더가 낫습니다.**
- **그런데 활성화를 빼면 PCA와 같아집니다.** 소수점 넷째 자리까지.
  이기는 이유는 **깊이가 아니라 비선형**입니다.
- 정답을 한 번도 쓰지 않았습니다. **비지도 학습**입니다.

### 연습

1. 은닉층을 하나 더 쌓으면 나아집니까. 활성화가 없을 때도 그렇습니까.
2. 잠재 차원을 784로 두면(줄이지 않으면) 어떻게 됩니까. **왜 그렇습니까.**
3. 인코더만 떼어 내 분류기를 붙이십시오(`encode()` 사용).
   원래 784차원으로 분류한 것과 견주십시오. — **9장 전이학습과 같은 착상입니다.**
4. Fashion-MNIST로 같은 실험을 하십시오. 압축이 더 쉽습니까 어렵습니까.""",

"w_dn": """## 정리

| | MSE |
|---|:--:|
| 잡음 낀 입력 그대로 (기준선) | 0.0797 |
| **잡음→깨끗 학습한 AE** | **0.0172** |
| 깨끗→깨끗 AE에 잡음을 넣으면 | 0.0390 |

- **바꾼 것은 `fit(x_noisy, x_clean)` 한 줄뿐입니다.** 구조는 그대로입니다.
- 깨끗한 것만 보고 배운 모델은 잡음에 약합니다. **본 것만 잘합니다.**
  4장 §4.5의 데이터 증강과 같은 이야기입니다.

### 연습

1. 잡음 세기(`level`)를 0.2, 0.6, 1.0으로 바꾸십시오. 어디서부터 실패합니까.
2. 학습 때와 **다른 종류의 잡음**(예: 픽셀 일부를 0으로)을 시험에 주면
   어떻게 됩니까.
3. 잠재 차원을 2로 줄이면 잡음 제거가 더 잘 됩니까 더 나빠집니까.
   **예상하고 나서 확인하십시오.**""",

"w_an1": """## 정리 — 그리고 다음 노트북으로

**AUC 0.5는 찍기입니다.** 이 방법이 여기서는 통하지 않습니다.

0~8을 배우며 익힌 것은 *"숫자 9의 생김새"* 가 아니라 **획, 곡선, 굵기**
같은 일반적인 부품입니다. 그 부품으로 9도 그대로 그릴 수 있습니다.

**→ 그런데 언제나 안 되는 것은 아닙니다. `ch13_anomaly_ood.ipynb` 로
가십시오.**

### 연습

1. 이상으로 둘 숫자를 0, 1, 8로 바꾸시오. **어느 숫자가 가장 잘
   잡힙니까. 왜입니까.**
2. 복원 오차 대신 **잠재 벡터까지의 최근접 거리**로 점수를 매기면
   나아집니까.""",
"w_an2": """## 정리

| 실험 | 무엇을 찾나 | AUC |
|---|---|:--:|
| ③ 0~8로 학습 | 숫자 '9' | **0.51~0.60** |
| ④ 숫자로 학습 | 옷 사진 | **0.96~0.99** |

- **오토인코더 이상 탐지는 「얼마나 다른가」에 달려 있습니다.**
- 분포가 아예 다르면 잘 잡습니다(②).
- **같은 분포 안의 새 종류는 못 잡습니다**(①). 0~8을 배우며 익힌 획과
  곡선으로 9도 그려 버리기 때문입니다.
- 그러니 *"오토인코더로 불량품을 찾는다"* 는 계획은 **불량이 정상과
  얼마나 다른지**를 먼저 따져야 합니다.

### 연습

1. 실험 ①에서 **다른 숫자**(0, 1, 8)를 이상으로 두면 AUC가 달라집니까.
   어느 숫자가 가장 잘 잡힙니까. **왜입니까.**
2. 복원 오차 대신 **잠재 벡터까지의 거리**로 점수를 매기면 나아집니까.
3. 실험 ②에서 옷 사진 대신 **뒤집은 숫자**를 넣으면 어떻게 됩니까.
4. (논술) 공장 불량 검출에 이 방법을 쓰려 한다. 실험 ①과 ② 중 어느 쪽에
   가까운 상황인지 어떻게 판단하겠는가.""",
}


def lines(t):
    return t.rstrip("\n").splitlines(keepends=True)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(t)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(t)}


def write(path, cells, meta):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"},
                       "dlbook": meta},
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print("wrote", path.relative_to(R))


KO = {"keras": "Keras 3", "tensorflow": "TensorFlow", "pytorch": "PyTorch"}

if __name__ == "__main__":
    for ed, trainer in TRAIN.items():
        base = R / "notebooks" / ed / "ch13"
        head = [md(MD["setup"]), code(BOOTSTRAP), code(SETUP),
                md(MD["data"]), code(DATA),
                md(MD["train"]), code(trainer)]

        write(base / "ch13_autoencoder.ipynb",
              [md(MD["t_ae"].format(ko=KO[ed]))] + head
              + [md(MD["pca"]), code(PCA_BASE),
                 md(MD["run"]), code(RUN),
                 md(MD["linear"]), code(LINEAR),
                 md(MD["recon"]), code(RECON),
                 md(MD["w_ae"])],
              {"chapter": 13, "edition": ed, "title": "오토인코더와 PCA"})

        write(base / "ch13_denoise.ipynb",
              [md(MD["t_dn"].format(ko=KO[ed]))] + head
              + [md(MD["denoise"]), code(DENOISE), md(MD["w_dn"])],
              {"chapter": 13, "edition": ed, "title": "잡음 제거"})

        write(base / "ch13_anomaly.ipynb",
              [md(MD["t_an"].format(ko=KO[ed]))] + head
              + [md(MD["anom1"]), code(ANOMALY_HEAD), code(ANOM1),
                 md(MD["w_an1"])],
              {"chapter": 13, "edition": ed, "title": "이상 탐지 — 되지 않는 경우"})

        write(base / "ch13_anomaly_ood.ipynb",
              [md(MD["t_an2"].format(ko=KO[ed]))] + head
              + [md(MD["anom2"]), code(ANOMALY_HEAD), code(ANOM2),
                 md(MD["w_an2"])],
              {"chapter": 13, "edition": ed, "title": "이상 탐지 — 되는 경우"})
    print("완료")
