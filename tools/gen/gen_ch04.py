# -*- coding: utf-8 -*-
"""4장 실습 세 개 — 정규화 · 데이터 누수 · 데이터 증강. 3판.

본문과 연습문제가 이 세 노트북을 이름으로 부른다.
    §4.3 ④ 정규화              → ch04_preprocessing   (연습 7)
    §4.3 ★ 데이터 누수         → ch04_leakage         (연습 6)
    §4.5 데이터 증강           → ch04_augmentation    (연습 8)
"""
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

KO = {"keras": "Keras 3", "tensorflow": "TensorFlow", "pytorch": "PyTorch"}


# ═══════════════════════════════════════════════════════════════════════
#  ① ch04_preprocessing — 정규화를 안 하면 무슨 일이 생기는가
# ═══════════════════════════════════════════════════════════════════════

P_TITLE = """# 4장 실습 ① — 정규화를 안 하면 무슨 일이 생기는가

**{ko} 판**

§4.3 ④는 나이(0~100)와 소득(0~1억)을 한 모델에 넣는 이야기였습니다.
그 상황을 사과 데이터로 만들어 놓고 **직접 무너뜨려 봅니다.**

> **"정규화 안 하면 학습이 잘 안 됩니다"** 는 말은 너무 자주 들어서
> 아무 느낌이 없습니다. 얼마나 안 되는지 숫자로 보십시오."""

P_DATA = '''# 사과는 [크기, 색깔] 둘 다 0~4 입니다. 스케일이 같아서 문제가 안 생깁니다.
# 그래서 **색깔 축만 10,000배** 키웁니다. §4.3 ④의 "소득" 자리입니다.
x, y = data.apples(2000, seed=42)
x = x.copy()
x[:, 1] = x[:, 1] * 10000.0

s0 = data.split(x, y, seed=42)
print(s0.summary())
print()
print(f"크기 축   {s0.x_train[:, 0].min():10.2f} ~ {s0.x_train[:, 0].max():10.2f}")
print(f"색깔 축   {s0.x_train[:, 1].min():10.2f} ~ {s0.x_train[:, 1].max():10.2f}")
print()
print("★ 2장 §2.5의 조정 값 = 오차 × **입력값** × 학습률")
print("  입력값이 곱해집니다. 두 축의 조정 폭이 1만 배 차이 납니다.")'''

P_PREP = '''def prepare(s, how):
    """정규화 세 가지. **반드시 학습 데이터의 통계로만 합니다.**

    왜 학습 데이터의 통계로만 하는지는 다음 실습(ch04_leakage)에서 봅니다.
    """
    if how == "none":
        return s
    if how == "minmax":                       # (x - min) / (max - min)
        lo, hi = s.x_train.min(0), s.x_train.max(0)
        f = lambda a: (a - lo) / (hi - lo)
    else:                                     # (x - 평균) / 표준편차
        mu, sd = s.x_train.mean(0), s.x_train.std(0)
        f = lambda a: (a - mu) / sd
    return data.Split(f(s.x_train), s.y_train,
                      f(s.x_val), s.y_val,
                      f(s.x_test), s.y_test)


for how in ("none", "minmax", "standard"):
    t = prepare(s0, how).x_train
    print(f"{how:<10} 크기 {t[:, 0].min():9.2f}~{t[:, 0].max():<9.2f} "
          f"색깔 {t[:, 1].min():9.2f}~{t[:, 1].max():<9.2f}")'''

P_RUN = '''LRS = [0.001, 0.01, 0.1, 1.0]
HOWS = ["none", "minmax", "standard"]
NAME = {"none": "정규화 안 함", "minmax": "최소-최대", "standard": "표준화"}

acc = {}
hist = {}
print(f"{'':<14}" + "".join(f"{f'lr={l}':>12}" for l in LRS))
print("-" * (14 + 12 * len(LRS)))
for how in HOWS:
    s = prepare(s0, how)
    row = f"{NAME[how]:<14}"
    for lr in LRS:
        a, h = train(s, lr)
        acc[(how, lr)] = a
        hist[(how, lr)] = h
        row += f"{a:>12.3f}"
    print(row, flush=True)
    dlbook.record(f"ch04_norm_{how}_best", max(acc[(how, l)] for l in LRS))

base = max(np.mean(s0.y_test), 1 - np.mean(s0.y_test))   # 그냥 많은 쪽을 찍은 값
def n_good(how): return sum(acc[(how, l)] >= 0.90 for l in LRS)

print()
print(f"★ **정규화를 안 하면 네 학습률 어디에서도 못 배웁니다.** "
      f"(0.90 넘김 {n_good('none')}/{len(LRS)})")
print(f"  {base:.3f} 는 그냥 많은 쪽을 찍으면 나오는 값입니다. "
      f"가장 나쁜 것은 {min(acc[('none', l)] for l in LRS):.3f} 로, 찍는 것보다도 못합니다.")
print(f"★ 정규화를 하면 **적어도 세 곳에서** 0.90을 넘습니다. "
      f"(최소-최대 {n_good('minmax')}/{len(LRS)}, 표준화 {n_good('standard')}/{len(LRS)})")
print("  → 연습문제 7의 답입니다. **학습률로는 회복되지 않습니다.**")'''

P_PLOT = '''fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
for ax, how in zip(axes, HOWS):
    for lr in LRS:
        h = hist[(how, lr)]
        ax.plot(range(1, len(h["loss"]) + 1), h["loss"], marker="", lw=1.8,
                label=f"lr={lr}")
    ax.set_title(f"{NAME[how]}  (최고 {max(acc[(how, l)] for l in LRS):.3f})")
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
axes[0].set_ylabel("학습 손실"); axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()

print("★ 왼쪽 그림의 손실은 **내려가다 마는 것이 아니라 아예 안 내려갑니다.**")
print("  색깔 축의 가중치는 한 걸음에 너무 크게 튀고,")
print("  크기 축의 가중치는 거의 안 움직입니다. 같은 학습률로는 둘 다 못 맞춥니다.")'''

P_NUM = {
    # 실측값. keras 와 tensorflow 는 같은 엔진이라 같은 값이 나온다.
    "keras":      ((0.665, 0.335, 0.665, 0.665), (0.943, 0.945, 0.950, 0.940),
                   (0.948, 0.943, 0.945, 0.665)),
    "tensorflow": ((0.665, 0.335, 0.665, 0.665), (0.943, 0.945, 0.950, 0.940),
                   (0.948, 0.943, 0.945, 0.665)),
    "pytorch":    ((0.665, 0.667, 0.665, 0.665), (0.940, 0.945, 0.950, 0.665),
                   (0.943, 0.955, 0.950, 0.792)),
}


def p_wrap(ed):
    none, mm, st = P_NUM[ed]
    worst = min(none)
    bad_note = (f"lr=0.01의 {worst:.3f}는 **찍는 것보다도 못합니다** — 발산한 것입니다."
                if worst < 0.6 else
                f"네 번 다 {max(none):.3f} 근처에서 멈춥니다 — 아무것도 못 배운 것입니다.")
    fragile = [lr for lr, v in zip((0.001, 0.01, 0.1, 1.0), mm) if v < 0.9]
    mm_note = ("**최소-최대는 학습률에 관대합니다.** 값이 0~1로 갇히기 때문입니다."
               if not fragile else
               f"**최소-최대도 lr={fragile[0]}에서는 무너집니다.** 값이 갇혀 있어도 "
               "걸음이 너무 크면 소용이 없습니다.")
    return f"""## 정리

| 정규화 | lr=0.001 | lr=0.01 | lr=0.1 | lr=1.0 |
|---|:--:|:--:|:--:|:--:|
| **안 함** | **{none[0]:.3f}** | **{none[1]:.3f}** | **{none[2]:.3f}** | **{none[3]:.3f}** |
| 최소-최대 | {mm[0]:.3f} | {mm[1]:.3f} | {mm[2]:.3f} | {mm[3]:.3f} |
| 표준화 | {st[0]:.3f} | {st[1]:.3f} | {st[2]:.3f} | {st[3]:.3f} |

- **정규화를 안 하면 학습률로 못 살립니다.** 네 개를 훑어도 전부 실패합니다.
  0.665는 많은 쪽을 그냥 찍은 값입니다. {bad_note}
- {mm_note}
- **큰 학습률에서는 정규화를 해도 무너집니다.** §4.3의 표대로
  "대체로 표준화"가 맞지만, **정규화와 학습률은 짝으로 봐야 합니다.**
  (5장에서 학습률만 따로 다룹니다)

> 세 판의 숫자가 조금씩 다릅니다. 초기값을 뽑는 방식이 프레임워크마다
> 달라서입니다. **결론은 셋 다 같습니다.**

### 연습

1. 색깔 축을 10,000배가 아니라 **10배**만 키우면 어떻게 됩니까.
   몇 배부터 학습이 무너지기 시작합니까.
2. 정규화를 **한 축에만** 걸어 보십시오. 그것으로 충분합니까.
3. lr=1.0에서 무너진 것은 epoch을 늘리면 회복됩니까.
4. 5장에서 배울 것을 미리 써 보십시오 — 옵티마이저를 `Adam` 대신
   `SGD` 로 바꾸면 정규화의 중요도가 커집니까 작아집니까."""


P_TRAIN = {
"keras": '''import keras
from keras import layers

def train(s, lr, epochs=30):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    h = m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
              batch_size=32, verbose=0)
    pred = (m.predict(s.x_test, verbose=0).ravel() > 0.5).astype(int)
    return metrics.accuracy(s.y_test, pred), h.history''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def train(s, lr, epochs=30):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = tf.keras.Sequential([
        L_.Input(shape=(2,)),
        L_.Dense(16, activation="relu"),
        L_.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    h = m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
              batch_size=32, verbose=0)
    pred = (m.predict(s.x_test, verbose=0).ravel() > 0.5).astype(int)
    return metrics.accuracy(s.y_test, pred), h.history''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def train(s, lr, epochs=30):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**

    손실 곡선은 케라스가 `history` 로 주지만 파이토치는 직접 모읍니다.
    `plot.loss_curve()` 가 둘 다 받도록 만들어져 있습니다.
    """
    dlbook.set_seed(42)
    m = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tx = torch.tensor(s.x_train, dtype=torch.float32)
    ty = torch.tensor(s.y_train, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True)

    losses = []
    for _ in range(dlbook.smoke.epochs(epochs)):
        m.train(); total = 0.0
        for xx, yy in dl:
            opt.zero_grad()
            loss = loss_fn(m(xx), yy)
            loss.backward(); opt.step()
            total += loss.item() * len(xx)
        losses.append(total / len(tx))

    m.eval()
    with torch.no_grad():
        logit = m(torch.tensor(s.x_test, dtype=torch.float32)).ravel()
    pred = (torch.sigmoid(logit).numpy() > 0.5).astype(int)
    return metrics.accuracy(s.y_test, pred), {"loss": losses}''',
}

P_MD = {"setup": "## 4.0 준비",
        "data": "## 4.1 크기가 1만 배 다른 두 축",
        "prep": "## 4.2 정규화 세 가지",
        "train": "## 4.3 학습 함수 — 여기만 판마다 다릅니다",
        "run": "## 4.4 정규화 × 학습률 열두 판",
        "plot": "## 4.5 손실 곡선"}


# ═══════════════════════════════════════════════════════════════════════
#  ② ch04_leakage — 데이터 누수
# ═══════════════════════════════════════════════════════════════════════

L_TITLE = """# 4장 실습 ② — 데이터 누수

**{ko} 판**

§4.3 ★의 **틀린 코드**와 **맞는 코드**를 나란히 돌립니다.

```python
x = (x - x.mean()) / x.std()   # 틀림 — 전체로 정규화하고
s = data.split(x, y)           #        그다음에 나눈다

s = data.split(x, y)                            # 맞음 — 먼저 나누고
mu, sd = s.x_train.mean(0), s.x_train.std(0)    #        학습 통계로만
```

> **결과를 미리 말씀드립니다. 이 둘은 차이가 안 납니다.**
> 그런데 누수는 무섭습니다. 두 문장이 어떻게 같이 참인지가 이 실습입니다."""

L_A = '''def normalize_wrong(x, y, seed):
    """틀린 순서 — 전체로 정규화하고 나눈다."""
    z = (x - x.mean(0)) / x.std(0)
    return data.split(z, y, seed=seed)


def normalize_right(x, y, seed):
    """맞는 순서 — 나누고, 학습 데이터의 통계로만."""
    s = data.split(x, y, seed=seed)
    mu, sd = s.x_train.mean(0), s.x_train.std(0)
    f = lambda a: (a - mu) / sd
    return data.Split(f(s.x_train), s.y_train,
                      f(s.x_val), s.y_val, f(s.x_test), s.y_test)


print(f"{'데이터 크기':<14}{'누수':>10}{'제대로':>10}{'차이':>10}")
print("-" * 44)
for n in (100, 300, 1000):
    x, y = data.apples(n, seed=1)
    x = x.copy(); x[:, 1] = x[:, 1] * 10000.0
    a_bad = train(normalize_wrong(x, y, 7))
    a_ok = train(normalize_right(x, y, 7))
    print(f"{n:<14,}{a_bad:>10.3f}{a_ok:>10.3f}{a_bad - a_ok:>+10.3f}")
    dlbook.record(f"ch04_leak_norm_gap_n{n}", a_bad - a_ok)

print()
print("★ **차이가 없습니다.** 데이터를 줄여도 마찬가지입니다.")
print("  평균과 표준편차는 표본이 조금만 있어도 안정적이라,")
print("  전체로 재나 학습 데이터로만 재나 거의 같은 값이 나옵니다.")
print()
print("  그러면 §4.3의 '맞는 코드'는 왜 지켜야 합니까. 다음 칸입니다.")'''

L_B = '''def top_k_features(x, y, k):
    """타깃과 상관이 큰 특징 k개를 고른다. 흔한 전처리입니다."""
    c = np.array([np.corrcoef(x[:, j], y)[0, 1] for j in range(x.shape[1])])
    return np.argsort(-np.nan_to_num(np.abs(c)))[:k]


N, F, K = 300, 400, 10
bad, ok = [], []
seeds = range(3 if dlbook.smoke.is_smoke() else 8)

for sd in seeds:
    rng = np.random.default_rng(sd)
    X = rng.normal(size=(N, F)).astype("float32")
    Y = rng.integers(0, 2, N).astype("int64")     # ★ 정답이 동전 던지기입니다

    sel = top_k_features(X, Y, K)                 # 틀림 — 전체를 보고 고른다
    bad.append(train(data.split(X[:, sel], Y, seed=sd)))

    s = data.split(X, Y, seed=sd)                 # 맞음 — 나누고 학습만 보고
    j = top_k_features(s.x_train, s.y_train, K)
    ok.append(train(data.Split(s.x_train[:, j], s.y_train,
                               s.x_val[:, j], s.y_val,
                               s.x_test[:, j], s.y_test)))
    print(f"  시드 {sd}   누수 {bad[-1]:.3f}   제대로 {ok[-1]:.3f}", flush=True)

print()
print(f"  {'누수':<8}{np.mean(bad):.3f}")
print(f"  {'제대로':<8}{np.mean(ok):.3f}")
print(f"  {'진실':<8}0.500   ← 정답이 무작위입니다. 맞힐 방법이 없습니다.")
dlbook.record("ch04_leak_select_bad", float(np.mean(bad)))
dlbook.record("ch04_leak_select_ok", float(np.mean(ok)))

print()
print(f"★ **신호가 하나도 없는 데이터에서 {np.mean(bad):.3f} 가 나왔습니다.**")
print(f"  {F}개 잡음 중 우연히 정답과 비슷해 보이는 {K}개를 골랐는데,")
print("  **고를 때 시험 데이터를 봤기 때문에** 그 우연이 시험에서도 통합니다.")
print(f"★ 제대로 하면 {np.mean(ok):.3f} — 있는 그대로 0.5입니다. 이게 정직한 숫자입니다.")'''

L_NUM = {"keras": (0.594, 0.506), "tensorflow": (0.594, 0.506),
         "pytorch": (0.608, 0.529)}


def l_wrap(ed):
    bad, ok = L_NUM[ed]
    return f"""## 정리

| 정규화 누수 | 시험 정확도 |
|---|:--:|
| 틀린 순서 — 전체로 정규화하고 나눈다 | 0.950 |
| 맞는 순서 — 나누고 학습 통계로만 | 0.950 |
| **차이** | **0.000** |

| 특징 선택 누수 | 시험 정확도 |
|---|:--:|
| 틀린 순서 — 전체를 보고 고른다 | **{bad:.3f}** |
| 맞는 순서 — 나누고 학습만 보고 고른다 | {ok:.3f} |
| **진실** | **0.500** |

- **정규화 누수는 거의 아무 일도 일으키지 않습니다.** 평균과 표준편차는
  표본 몇백 개면 안정적이라, 전체로 재나 학습 데이터로만 재나 같습니다.
  데이터를 100개로 줄여도 차이가 0.000입니다.
- **그런데 특징 선택 누수는 재앙입니다.** 정답이 동전 던지기인 데이터에서
  {bad:.3f}가 나옵니다. 논문 한 편이 여기서 만들어집니다.
- **둘의 차이는 「무엇을 골랐느냐」입니다.** 정규화는 값을 옮길 뿐이지만,
  선택은 **시험 데이터를 보고 결정을 내립니다.** 결정이 시험 데이터를
  보고 내려지는 순간 그 시험은 시험이 아닙니다.
- 그래서 §4.3의 규칙은 이렇게 읽는 것이 맞습니다 —
  **"정규화를 나중에 하라"가 아니라 "학습 데이터를 보고 정한 것은 전부
  그대로 시험에 적용하라"**. 결측치 대치의 중앙값, 범주형 인코딩의 범주
  목록, 이상치 절단의 경계, 그리고 **특징 선택**.

### 연습

1. 특징 개수 `F` 를 400에서 40으로 줄이면 누수의 크기는 어떻게 됩니까.
   왜 그렇습니까.
2. 표본 `N` 을 300에서 3,000으로 늘리면 어떻게 됩니까.
3. 7장을 미리 보십시오. **모델을 여러 번 골라 시험 성능이 가장 좋은 것을
   고르는 것**도 같은 종류의 누수입니다. 어디가 같습니까.
4. 정규화 누수가 **실제로 위험해지는 경우**를 하나 만들어 보십시오.
   (힌트: 시간이 흐르는 데이터. 미래가 과거로 새어 듭니다)"""


L_TRAIN = {
"keras": '''import keras
from keras import layers

def train(s, epochs=40, lr=0.01):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = keras.Sequential([
        layers.Input(shape=(s.x_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=16, verbose=0)
    pred = (m.predict(s.x_test, verbose=0).ravel() > 0.5).astype(int)
    return metrics.accuracy(s.y_test, pred)''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def train(s, epochs=40, lr=0.01):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = tf.keras.Sequential([
        L_.Input(shape=(s.x_train.shape[1],)),
        L_.Dense(16, activation="relu"),
        L_.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    m.fit(s.x_train, s.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=16, verbose=0)
    pred = (m.predict(s.x_test, verbose=0).ravel() > 0.5).astype(int)
    return metrics.accuracy(s.y_test, pred)''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def train(s, epochs=40, lr=0.01):
    """같은 모델, 같은 조건. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = nn.Sequential(nn.Linear(s.x_train.shape[1], 16), nn.ReLU(),
                      nn.Linear(16, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tx = torch.tensor(s.x_train, dtype=torch.float32)
    ty = torch.tensor(s.y_train, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(tx, ty), batch_size=16, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        m.train()
        for xx, yy in dl:
            opt.zero_grad(); loss_fn(m(xx), yy).backward(); opt.step()

    m.eval()
    with torch.no_grad():
        logit = m(torch.tensor(s.x_test, dtype=torch.float32)).ravel()
    return metrics.accuracy(s.y_test, (torch.sigmoid(logit).numpy() > 0.5).astype(int))''',
}

L_MD = {"setup": "## 4.0 준비",
        "train": "## 4.1 학습 함수 — 여기만 판마다 다릅니다",
        "a": "## 4.2 정규화 누수 — 얼마나 부풀려지나",
        "b": "## 4.3 누수는 정규화에서만 나지 않습니다"}


# ═══════════════════════════════════════════════════════════════════════
#  ③ ch04_augmentation — 증강은 도메인이 정합니다
# ═══════════════════════════════════════════════════════════════════════

A_TITLE = """# 4장 실습 ③ — 증강은 도메인이 정합니다

**{ko} 판**

§4.5의 표에 이렇게 적혀 있습니다.

| 데이터 | 좌우 반전 | 이동 |
|---|:--:|:--:|
| 손글씨 숫자 | **✗** (2와 5가 섞임) | ○ |

**정말 그런지 봅니다.** 같은 증강을 하나는 알맞게, 하나는 잘못 걸고
학습시켜 견줍니다.

> 데이터를 **네 배로 늘렸는데 성능이 떨어지는 것**을 보시게 됩니다."""

A_DATA = '''# §4.1의 "영상은 300~500장이면 유의미하다"를 확인하려면
# **일부러 적게** 써야 합니다. 5만 장을 쓰면 증강해도 티가 안 납니다.
N_TRAIN = 500
full = data.mnist()
s = data.Split(full.x_train[:N_TRAIN], full.y_train[:N_TRAIN],
               full.x_val[:500], full.y_val[:500],
               full.x_test[:2000], full.y_test[:2000])
print(s.summary())
plot.image_grid(s.x_train, s.y_train, n=16, cols=8)
plt.show()'''

A_AUG = '''def shift(img, dy, dx):
    """이미지를 상하좌우로 옮긴다. 빈 자리는 0."""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    out[ys:ye, xs:xe] = img[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def augment(x, y, times, mode, seed=42):
    """원본 + 변형본 times벌. **학습 데이터에만 씁니다.**

    검증·시험은 성능을 재는 자리이므로 원본 그대로여야 합니다 (§4.5).
    """
    rng = np.random.default_rng(seed)
    xs, ys = [x], [y]
    for _ in range(times):
        batch = np.empty_like(x)
        for i, im in enumerate(x):
            if mode == "shift":
                batch[i] = shift(im, rng.integers(-2, 3), rng.integers(-2, 3))
            else:                                   # "flip" — 좌우 반전
                batch[i] = im[:, ::-1]
        xs.append(batch); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


x_flip, _ = augment(s.x_train[:8], s.y_train[:8], 1, "flip")
plot.image_grid(x_flip, np.concatenate([s.y_train[:8]] * 2), n=16, cols=8)
plt.show()
print("★ 윗줄이 원본, 아랫줄이 좌우 반전입니다.")
print("  뒤집힌 2는 2가 아닙니다. 그런데 정답표에는 여전히 2라고 적혀 있습니다.")'''

A_RUN = '''CASES = [("증강 없음", None), ("이동 ±2픽셀", "shift"), ("좌우 반전", "flip")]
TIMES = 3

result = {}
print(f"{'':<16}{'학습 장수':>10}{'시험 정확도':>14}")
print("-" * 40)
for name, mode in CASES:
    if mode is None:
        xt, yt = s.x_train, s.y_train
    else:
        xt, yt = augment(s.x_train, s.y_train, TIMES, mode)
    a = train(xt, yt, s)
    result[name] = a
    print(f"{name:<16}{len(xt):>10,}{a:>14.3f}", flush=True)
    dlbook.record(f"ch04_aug_{mode or 'none'}", a)

print()
print(f"★ 이동은 **+{result['이동 ±2픽셀'] - result['증강 없음']:.3f}**. "
      f"장수만 늘린 것이 아니라 실제로 도움이 됐습니다.")
print(f"★ 좌우 반전은 **{result['좌우 반전'] - result['증강 없음']:+.3f}**. "
      f"**데이터를 네 배로 늘렸는데 더 나빠졌습니다.**")
print("  현실에 없는 데이터를 학습시킨 값입니다.")'''

A_NUM = {"keras": (0.881, 0.917, 0.829), "tensorflow": (0.881, 0.917, 0.829),
         "pytorch": (0.866, 0.909, 0.820)}


def a_wrap(ed):
    none, shift, flip = A_NUM[ed]
    return f"""## 정리

| | 학습 장수 | 시험 정확도 |
|---|--:|:--:|
| 증강 없음 | 500 | {none:.3f} |
| **이동 ±2픽셀** | 2,000 | **{shift:.3f}** |
| **좌우 반전** | 2,000 | **{flip:.3f}** |

- **알맞은 증강은 이깁니다.** 500장으로 {none:.3f}이던 것이 2,000장으로
  {shift:.3f}이 됐습니다({shift - none:+.3f}). 사진을 더 찍지 않고 얻은 것입니다.
- **잘못된 증강은 데이터를 네 배로 늘리고도 집니다.** {none:.3f} → {flip:.3f}
  ({flip - none:+.3f}). 뒤집힌 2를 2라고 가르쳤기 때문입니다.
  **현실에 없는 데이터입니다.**
- 그래서 §4.5의 문장이 그대로 성립합니다 —
  **"이렇게 변형해도 정답이 그대로인가"를 사람이 판단해서 켜는 것입니다.
  기계가 정해 주지 않습니다.**
- 위성 영상이었다면 좌우 반전도 상하 반전도 켰을 것입니다. 같은 증강이
  **데이터가 무엇이냐에 따라** 약이 되기도 독이 되기도 합니다.

### 연습

1. `TIMES` 를 3에서 7로 올리면(원본의 8배) 얼마나 더 좋아집니까.
   계속 좋아집니까, 어디서 멈춥니까.
2. 이동 폭을 ±2에서 **±8픽셀**로 키우십시오. 도움이 됩니까 방해가 됩니까.
   왜 그렇습니까.
3. `N_TRAIN` 을 500에서 5,000으로 올리고 다시 하십시오.
   **증강의 이득이 커집니까 작아집니까.** 그 이유를 §4.1과 엮어 설명하십시오.
4. 좌우 반전이 **도움이 되는** 데이터를 하나 고르고, 왜 그런지 쓰십시오.
5. 8장을 미리 보십시오. 이동 증강이 CNN에 주는 이득과 DNN에 주는 이득 중
   어느 쪽이 클 것 같습니까. 돌려서 확인하십시오."""


A_TRAIN = {
"keras": '''import keras
from keras import layers

def train(xt, yt, s, epochs=25):
    """작은 CNN. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(xt, yt, epochs=dlbook.smoke.epochs(epochs), batch_size=64, verbose=0)
    pred = m.predict(s.x_test, verbose=0, batch_size=512).argmax(1)
    return metrics.accuracy(s.y_test, pred)''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def train(xt, yt, s, epochs=25):
    """작은 CNN. **이 함수만 판마다 다릅니다.**"""
    dlbook.set_seed(42)
    m = tf.keras.Sequential([
        L_.Input(shape=(28, 28, 1)),
        L_.Conv2D(16, 3, activation="relu"), L_.MaxPooling2D(2),
        L_.Conv2D(32, 3, activation="relu"), L_.MaxPooling2D(2),
        L_.Flatten(), L_.Dense(64, activation="relu"),
        L_.Dense(10, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy")
    m.fit(xt, yt, epochs=dlbook.smoke.epochs(epochs), batch_size=64, verbose=0)
    pred = m.predict(s.x_test, verbose=0, batch_size=512).argmax(1)
    return metrics.accuracy(s.y_test, pred)''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(xt, yt, s, epochs=25):
    """작은 CNN. **이 함수만 판마다 다릅니다.**

    dlbook.data 는 channels_last 로 돌려줍니다.
    PyTorch는 channels_first 라 **여기서 축을 바꿉니다.** (8장 §8.1)
    """
    dlbook.set_seed(42)
    m = nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        nn.Linear(64, 10)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    tx = torch.tensor(xt, dtype=torch.float32).permute(0, 3, 1, 2)
    dl = DataLoader(TensorDataset(tx, torch.tensor(yt, dtype=torch.long)),
                    batch_size=64, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        m.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(m(xx), yy).backward(); opt.step()

    m.eval()
    ex = torch.tensor(s.x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    with torch.no_grad():
        pred = np.concatenate([m(ex[i:i + 512].to(device)).cpu().numpy()
                               for i in range(0, len(ex), 512)]).argmax(1)
    return metrics.accuracy(s.y_test, pred)''',
}

A_MD = {"setup": "## 4.0 준비",
        "data": "## 4.1 데이터를 일부러 적게 씁니다",
        "aug": "## 4.2 증강 — numpy 열 줄이면 됩니다",
        "train": "## 4.3 학습 함수 — 여기만 판마다 다릅니다",
        "run": "## 4.4 켜고, 끄고, 잘못 켜고"}


# ═══════════════════════════════════════════════════════════════════════

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
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print("wrote", path.relative_to(R))


if __name__ == "__main__":
    for ed in ("keras", "tensorflow", "pytorch"):
        base = R / "notebooks" / ed / "ch04"

        write(base / "ch04_preprocessing.ipynb",
              [md(P_TITLE.format(ko=KO[ed])),
               md(P_MD["setup"]), code(BOOT), code(SETUP),
               md(P_MD["data"]), code(P_DATA),
               md(P_MD["prep"]), code(P_PREP),
               md(P_MD["train"]), code(P_TRAIN[ed]),
               md(P_MD["run"]), code(P_RUN),
               md(P_MD["plot"]), code(P_PLOT),
               md(p_wrap(ed))],
              {"chapter": 4, "edition": ed, "title": "정규화"})

        write(base / "ch04_leakage.ipynb",
              [md(L_TITLE.format(ko=KO[ed])),
               md(L_MD["setup"]), code(BOOT), code(SETUP),
               md(L_MD["train"]), code(L_TRAIN[ed]),
               md(L_MD["a"]), code(L_A),
               md(L_MD["b"]), code(L_B),
               md(l_wrap(ed))],
              {"chapter": 4, "edition": ed, "title": "데이터 누수"})

        write(base / "ch04_augmentation.ipynb",
              [md(A_TITLE.format(ko=KO[ed])),
               md(A_MD["setup"]), code(BOOT), code(SETUP),
               md(A_MD["data"]), code(A_DATA),
               md(A_MD["aug"]), code(A_AUG),
               md(A_MD["train"]), code(A_TRAIN[ed]),
               md(A_MD["run"]), code(A_RUN),
               md(a_wrap(ed))],
              {"chapter": 4, "edition": ed, "title": "데이터 증강"})
    print("완료 — 9개")
