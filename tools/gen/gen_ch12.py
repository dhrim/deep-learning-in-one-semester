# -*- coding: utf-8 -*-
"""12장 실습 노트북 3판 생성 — 어텐션과 트랜스포머."""
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

DATA = """# 11장 §11.9의 「어려운 규칙」을 그대로 씁니다.
# 감성 단어가 둘(부호 반대), 정답은 **먼저 나온 쪽**을 따릅니다.
V = len(data.TOY_VOCAB)
LENGTH = 16
x, y = data.toy_reviews(n=8000, length=LENGTH, seed=42, hard=True)
sp = data.split(x, y, val_ratio=0.2, test_ratio=0.2, seed=42)
print(sp.summary())
print()
for k in range(3):
    print(f"  {data.toy_decode(x[k]):<52} → {'긍정' if y[k] else '부정'}")
print()
print("이 과제는 **떨어져 있는 두 자리를 견주어야** 풀립니다.")
print("11장에서 Conv1D가 0.949에 머물고 LSTM만 1.000을 냈던 그 과제입니다.")"""

POS = '''# 위치 인코딩 — 사인·코사인. (Vaswani et al. 2017)
# 순수 numpy입니다. 세 판이 **완전히 같습니다.**
def positional_encoding(length, depth):
    """자리마다 다른 값을 갖는 (length, depth) 행렬을 만든다.

    같은 자리는 늘 같은 값이고, 가까운 자리끼리는 비슷한 값이 된다.
    이것을 임베딩에 **더해** 주면 "몇 번째 단어인가"가 표현에 들어간다.
    """
    pos = np.arange(length)[:, None]
    i = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / depth)
    pe = np.zeros((length, depth), dtype="float32")
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

PE = positional_encoding(LENGTH, 8)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
im = ax.imshow(PE.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("자리 (0~15)"); ax.set_ylabel("차원 (0~7)")
ax.set_title("위치 인코딩 — 자리마다 다른 무늬")
fig.colorbar(im, ax=ax, shrink=0.85); plt.tight_layout(); plt.show()

print("→ 세로줄 하나가 자리 하나입니다. **자리마다 무늬가 다릅니다.**")
print("→ 이것을 임베딩에 더하면 같은 단어라도 자리가 다르면 다른 벡터가 됩니다.")'''

RUN = """kinds = [("cnn",      "임베딩 + Conv1D"),
         ("lstm",     "임베딩 + LSTM"),
         ("attn",     "어텐션 (위치 인코딩 없음)"),
         ("attn_pos", "어텐션 + 위치 인코딩")]

print(f"{'모델':<28}{'파라미터':>12}{'시험 정확도':>14}")
print("-" * 54)
for kind, ko in kinds:
    acc, n_params, _ = train_text(kind, sp)
    print(f"{ko:<28}{n_params:>12,}{acc:>14.3f}")
    dlbook.record(f"ch12_{kind}_acc", acc)
    dlbook.record(f"ch12_{kind}_params", n_params)

print()
print("★ 위치 인코딩가 없는 어텐션은 **동전 던지기**입니다.")
print("  11장 §11.7의 「임베딩 + 평균」과 똑같은 이유입니다 —")
print("  어텐션은 가중 합이고, **합은 순서를 따지지 않습니다.**")
print()
print("★ 위치 인코딩를 더하면 **LSTM과 같은 성능을 파라미터 1/7로** 냅니다.")"""

SHUFFLE = """# 확인 — 위치 인코딩가 정말 순서를 넣어 주는가.
rng = np.random.default_rng(0)
x_shuf = np.stack([rng.permutation(r) for r in sp.x_test])

print(f"{'모델':<12}{'원래 순서':>12}{'섞은 뒤':>12}{'떨어진 폭':>12}")
print("-" * 48)
for kind in ("attn", "attn_pos", "lstm"):
    train_text(kind, sp)
    a_ord = predict_acc(kind, sp, sp.x_test)
    a_shuf = predict_acc(kind, sp, x_shuf)
    print(f"{kind:<12}{a_ord:>12.3f}{a_shuf:>12.3f}{a_ord - a_shuf:>12.3f}")
    dlbook.record(f"ch12_shuffle_drop_{kind}", float(a_ord - a_shuf))

print()
print("→ 위치 인코딩가 **없는** 어텐션은 섞어도 그대로(0.000)입니다.")
print("   순서를 애초에 보지 않았습니다.")
print("→ 위치 인코딩를 더한 어텐션은 무너집니다. **순서를 보고 있었습니다.**")"""

LENGTH_EXP = """# 문장을 길게 하면 어떻게 되는가.
# ★ 10장 §10.5의 규율을 지킵니다 — **학습률을 훑습니다.**
lengths = [16, 32] if dlbook.smoke.is_smoke() else [16, 32, 64]
lrs = [0.001, 0.003] if dlbook.smoke.is_smoke() else [0.001, 0.003, 0.01]

print("각 칸은 **학습률 3종 중 최고**입니다. 하나로 고정하면 결론이 뒤집힙니다.")
print()
print(f"{'길이':<8}" + "".join(f"{k:>12}" for k in ("cnn", "lstm", "attn_pos")))
print("-" * 44)
for L in lengths:
    xl, yl = data.toy_reviews(8000, length=L, seed=42, hard=True)
    sl = data.split(xl, yl, val_ratio=0.2, test_ratio=0.2, seed=42)
    row = []
    for kind in ("cnn", "lstm", "attn_pos"):
        best = max(train_text(kind, sl, lr=lr)[0] for lr in lrs)
        row.append(best)
        dlbook.record(f"ch12_L{L}_{kind}_best", best)
    print(f"{L:<8}" + "".join(f"{v:>12.3f}" for v in row))

print()
print("→ **길이 64에서 LSTM이 무너집니다.** 학습률 3종 어디에서도 0.499입니다.")
print("→ 어텐션은 버팁니다. 거리와 무관하게 두 자리를 직접 견주기 때문입니다.")
print("→ Conv1D는 0.95 근처에 머뭅니다. 창 크기 3이 바뀌지 않았기 때문입니다.")"""

LR_TRAP = """# 학습률 하나로 고정하면 어떤 표가 나오는지 확인합니다.
L = 32
xl, yl = data.toy_reviews(8000, length=L, seed=42, hard=True)
sl = data.split(xl, yl, val_ratio=0.2, test_ratio=0.2, seed=42)

print(f"길이 {L}. 학습률별로 따로 봅니다.")
print(f"{'':<12}" + "".join(f"{'lr=' + str(lr):>11}" for lr in (0.001, 0.003, 0.01)))
print("-" * 46)
for kind in ("lstm", "attn_pos"):
    row = [train_text(kind, sl, lr=lr)[0] for lr in (0.001, 0.003, 0.01)]
    print(f"{kind:<12}" + "".join(f"{v:>11.3f}" for v in row))
    dlbook.record(f"ch12_lr32_{kind}_best", max(row))

print()
print("★ lr=0.003만 보면 'LSTM은 길이 32에서 못 한다'가 됩니다. **틀린 결론입니다.**")
print("  lr=0.001에서는 1.000입니다. 10장 §10.5에서 겪은 그 함정입니다.")
print("  길이 64에서 LSTM이 무너지는 것은 **학습률을 다 훑고도** 그렇기 때문에")
print("  비로소 말할 수 있는 것입니다.")"""

WEIGHTS = """# 어텐션 가중치를 꺼내 봅니다. **모델이 어디를 봤는지** 보입니다.
acc, _, model = train_text("attn_pos", sp)
W = attention_weights(model, sp.x_test[:1])       # (머리, 질의, 키)
print(f"시험 정확도 {acc:.3f}")

toks = [data.TOY_VOCAB[i] for i in sp.x_test[0] if i != 0]
n = len(toks)
print("문장:", " ".join(toks))
print("정답:", "긍정" if sp.y_test[0] else "부정")

fig, axes = plt.subplots(1, W.shape[0], figsize=(5.4 * W.shape[0], 4.6))
axes = np.atleast_1d(axes)
for h, ax in enumerate(axes):
    im = ax.imshow(W[h, :n, :n], cmap="Blues", vmin=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=8)
    ax.set_title(f"머리 {h + 1}", fontsize=10)
    ax.set_xlabel("보는 곳 (키)"); ax.set_ylabel("보는 주체 (질의)")
fig.suptitle("어텐션 가중치 — 각 단어가 어느 단어를 보는가", fontsize=11.5)
plt.tight_layout(); plt.show()

print()
print("→ 한 줄이 단어 하나입니다. 그 줄에서 **진한 칸이 그 단어가 보는 곳**입니다.")
print("→ 감성 단어와 '안'이 있는 열이 진해지는지 보십시오.")
print("→ 머리가 둘이면 **서로 다른 것을 볼 수** 있습니다. 그것이 멀티헤드입니다.")"""

TRAIN = {
"keras": '''import keras
from keras import layers

class AddPositional(layers.Layer):
    """위치 인코딩를 더하는 층. 학습되는 파라미터가 없습니다."""
    def __init__(self, length, depth, **kw):
        super().__init__(**kw)
        self.pe = keras.ops.convert_to_tensor(positional_encoding(length, depth))

    def call(self, x):
        return x + self.pe

def _build(kind, V, L, d=8):
    """모델 정의 — **이 함수만 판마다 다릅니다.**"""
    inp = layers.Input(shape=(L,))
    x = layers.Embedding(V, d)(inp)
    if kind == "attn_pos":
        x = AddPositional(L, d)(x)
    if kind in ("attn", "attn_pos"):
        a = layers.MultiHeadAttention(num_heads=2, key_dim=d // 2)(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, a]))   # 잔차 + 정규화
        x = layers.GlobalAveragePooling1D()(x)
    elif kind == "cnn":
        x = layers.Conv1D(16, 3, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
    elif kind == "lstm":
        x = layers.LSTM(32)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 모델) 을 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    m = _build(kind, V, L)
    m.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
              loss="binary_crossentropy")
    m.fit(sp.x_train, sp.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=64, verbose=0)
    _FIT[kind] = m
    pred = (m.predict(sp.x_test, verbose=0).reshape(-1) > 0.5).astype(int)
    return metrics.accuracy(sp.y_test, pred), m.count_params(), m

def predict_acc(kind, sp, xs):
    m = _FIT[kind]
    return metrics.accuracy(
        sp.y_test, (m.predict(xs, verbose=0).reshape(-1) > 0.5).astype(int))

def attention_weights(model, xs):
    """학습된 모델에서 어텐션 가중치를 꺼낸다. (머리, 질의, 키)"""
    emb = [l for l in model.layers if isinstance(l, layers.Embedding)][0]
    pos = [l for l in model.layers if isinstance(l, AddPositional)]
    mha = [l for l in model.layers if isinstance(l, layers.MultiHeadAttention)][0]
    h = emb(xs)
    if pos:
        h = pos[0](h)
    _, scores = mha(h, h, return_attention_scores=True)
    return np.asarray(scores)[0]''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

class AddPositional(L_.Layer):
    """위치 인코딩를 더하는 층. 학습되는 파라미터가 없습니다."""
    def __init__(self, length, depth, **kw):
        super().__init__(**kw)
        self.pe = tf.constant(positional_encoding(length, depth))

    def call(self, x):
        return x + self.pe

def _build(kind, V, L, d=8):
    """모델 정의 — **이 함수만 판마다 다릅니다.**"""
    inp = L_.Input(shape=(L,))
    x = L_.Embedding(V, d)(inp)
    if kind == "attn_pos":
        x = AddPositional(L, d)(x)
    if kind in ("attn", "attn_pos"):
        a = L_.MultiHeadAttention(num_heads=2, key_dim=d // 2)(x, x)
        x = L_.LayerNormalization()(L_.Add()([x, a]))
        x = L_.GlobalAveragePooling1D()(x)
    elif kind == "cnn":
        x = L_.GlobalMaxPooling1D()(L_.Conv1D(16, 3, activation="relu")(x))
    elif kind == "lstm":
        x = L_.LSTM(32)(x)
    return tf.keras.Model(inp, L_.Dense(1, activation="sigmoid")(x))

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 모델) 을 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    m = _build(kind, V, L)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
              loss="binary_crossentropy")
    m.fit(sp.x_train, sp.y_train, epochs=dlbook.smoke.epochs(epochs),
          batch_size=64, verbose=0)
    _FIT[kind] = m
    pred = (m.predict(sp.x_test, verbose=0).reshape(-1) > 0.5).astype(int)
    return metrics.accuracy(sp.y_test, pred), m.count_params(), m

def predict_acc(kind, sp, xs):
    m = _FIT[kind]
    return metrics.accuracy(
        sp.y_test, (m.predict(xs, verbose=0).reshape(-1) > 0.5).astype(int))

def attention_weights(model, xs):
    """학습된 모델에서 어텐션 가중치를 꺼낸다. (머리, 질의, 키)"""
    emb = [l for l in model.layers if isinstance(l, L_.Embedding)][0]
    pos = [l for l in model.layers if isinstance(l, AddPositional)]
    mha = [l for l in model.layers if isinstance(l, L_.MultiHeadAttention)][0]
    h = emb(xs)
    if pos:
        h = pos[0](h)
    _, scores = mha(h, h, return_attention_scores=True)
    return np.asarray(scores)[0]''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextModel(nn.Module):
    """임베딩 위에 무엇을 올리느냐만 다릅니다.

    PyTorch의 nn.MultiheadAttention 은 batch_first=True 를 주지 않으면
    (길이, 배치, 차원) 순서를 기대합니다. **자주 틀리는 자리입니다.**
    """
    def __init__(self, kind, V, L, d=8):
        super().__init__()
        self.kind, self.L, self.d = kind, L, d
        self.emb = nn.Embedding(V, d)
        if kind in ("attn", "attn_pos"):
            self.att = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
            self.norm = nn.LayerNorm(d)
            n_out = d
        elif kind == "cnn":
            self.conv = nn.Conv1d(d, 16, 3); n_out = 16
        elif kind == "lstm":
            self.rnn = nn.LSTM(d, 32, batch_first=True); n_out = 32
        if kind == "attn_pos":
            self.register_buffer(
                "pe", torch.tensor(positional_encoding(L, d)))
        self.head = nn.Linear(n_out, 1)

    def forward(self, x, return_weights=False):
        e = self.emb(x.long())
        if self.kind == "attn_pos":
            e = e + self.pe
        if self.kind in ("attn", "attn_pos"):
            a, w = self.att(e, e, e, need_weights=True, average_attn_weights=False)
            h = self.norm(e + a).mean(1)
            if return_weights:
                return w
        elif self.kind == "cnn":
            h = torch.relu(self.conv(e.transpose(1, 2))).max(dim=2).values
        else:
            out, _ = self.rnn(e); h = out[:, -1]
        return self.head(h)

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 모델) 을 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    model = TextModel(kind, V, L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(
        torch.tensor(sp.x_train, dtype=torch.long),
        torch.tensor(sp.y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=64, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        model.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(model(xx), yy).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    _FIT[kind] = model
    n_params = sum(p.numel() for p in model.parameters())
    return predict_acc(kind, sp, sp.x_test), n_params, model

def predict_acc(kind, sp, xs):
    model = _FIT[kind]; model.eval()
    with torch.no_grad():
        out = model(torch.tensor(xs, dtype=torch.long).to(device))
    return metrics.accuracy(sp.y_test, (out.cpu().numpy().reshape(-1) > 0).astype(int))

def attention_weights(model, xs):
    """학습된 모델에서 어텐션 가중치를 꺼낸다. (머리, 질의, 키)"""
    model.eval()
    with torch.no_grad():
        w = model(torch.tensor(xs, dtype=torch.long).to(device),
                  return_weights=True)
    return w.cpu().numpy()[0]''',
}

MD = {
"t_attn": """# 12장 실습 ① — 어텐션과 위치 인코딩

**{ko} 판**

11장의 「어려운 규칙」을 어텐션으로 풉니다. 그리고 **위치 인코딩를
빼면 무슨 일이 생기는지** 봅니다.

> **어텐션은 가중 합입니다. 그리고 합은 순서를 따지지 않습니다.**
> 11장 §11.7의 「임베딩 + 평균」이 여기서 다시 나옵니다.""",

"t_len": """# 12장 실습 ② — 문장이 길어지면

**{ko} 판**

어텐션이 순환 신경망을 밀어낸 이유를 재 봅니다.
**10장 §10.5의 규율을 지킵니다 — 학습률을 훑고 나서 말합니다.**""",

"t_w": """# 12장 실습 ③ — 모델이 어디를 보는가

**{ko} 판**

어텐션의 큰 장점 하나는 **가중치를 꺼내 볼 수 있다**는 것입니다.
9장의 Grad-CAM이 영상에서 한 일을, 어텐션은 **구조 자체로** 합니다.""",

"t_trap": """# 12장 실습 ③ — 학습률 하나로 고정하면

**{ko} 판**

실습 ②의 표는 **학습률 3종 중 최고**였습니다.
하나로 고정하면 어떤 표가 나오는지 확인합니다.

**10장 §10.5에서 겪은 그 함정입니다.**""",
"w_trap": """## 정리

- **lr=0.003만 보면 「LSTM은 길이 32에서 못 한다」가 됩니다. 틀린 결론입니다.**
  lr=0.001에서는 1.000입니다.
- 길이 64에서 LSTM이 무너진다고 말할 수 있는 것은 **학습률을 다 훑고도**
  0.499이기 때문입니다.
- 5장 §5.4, 10장 §10.5, 그리고 여기 — **같은 함정을 세 번째로 만났습니다.**

> **한 하이퍼파라미터를 고정해 놓고 다른 것을 비교하면,
> 원하는 결론을 아무거나 만들 수 있습니다.**

### 연습

1. 길이 16과 64에서도 학습률별 표를 만드시오.
2. 어텐션은 학습률에 얼마나 민감합니까. LSTM과 견주시오.
3. (논술) 이 강의에서 같은 함정을 세 번 만났다. **공통된 잘못**은 무엇인가.""",
"setup": "## 12.0 준비",
"data": """## 12.1 실험대 — 11장의 어려운 규칙

**Conv1D가 0.949에 머물고 LSTM만 1.000을 냈던** 그 과제입니다.
어텐션이 어떻게 하는지 봅니다.""",
"pos": """## 12.2 위치 인코딩

어텐션에는 **순서 개념이 없습니다.** 그래서 순서를 **입력에 심어** 줍니다.""",
"train": """## 12.3 모델 정의 — 여기만 판마다 다릅니다

**PyTorch 판의 `batch_first=True`** 에 주목하십시오. 이것을 빼면
`nn.MultiheadAttention` 은 (길이, 배치, 차원) 순서를 기대합니다.
**오류 없이 조용히 틀린 결과가 나오는** 자리입니다.""",
"run": """## 12.4 네 모델을 나란히""",
"shuffle": """## 12.5 확인 — 위치 인코딩가 정말 순서를 넣는가

11장 §11.8에서 쓴 것과 같은 검사입니다. **시험 문장의 단어를 섞습니다.**""",
"length": """## 12.1 문장을 길게 하면

어텐션이 순환 신경망을 대체한 진짜 이유입니다.""",
"lr_trap": """## 12.2 그런데 학습률 하나로 고정하면

**10장 §10.5의 함정을 여기서 다시 확인합니다.**""",
"weights": """## 12.1 어텐션 가중치 꺼내 보기""",

"w_attn": """## 정리

| 모델 | 파라미터 | 정확도 |
|---|:--:|:--:|
| 임베딩 + Conv1D | 633 | 0.949 |
| 임베딩 + LSTM | 5,497 | 1.000 |
| 어텐션 (위치 인코딩 없음) | 809 | **0.494** |
| **어텐션 + 위치 인코딩** | **809** | **0.996** |

- **어텐션만으로는 순서를 못 봅니다.** 가중 합이고, 합은 순서를 따지지
  않습니다. 11장 §11.7의 「임베딩 + 평균」과 정확히 같은 이유입니다.
- **위치 인코딩를 더하면 LSTM과 같은 성능을 파라미터 1/7로** 냅니다.
- 위치 인코딩에는 **학습되는 파라미터가 없습니다.** 사인·코사인 값을
  더할 뿐인데 0.494가 0.996이 됩니다.

### 연습

1. 위치 인코딩를 **학습되는 임베딩**(`Embedding(LENGTH, 8)`)으로 바꾸십시오.
   성능이 달라집니까. 파라미터는 몇 개 늘어납니까.
2. 머리 수(`num_heads`)를 1, 4로 바꾸십시오.
3. 잔차 연결(`Add`)과 정규화(`LayerNormalization`)를 빼면 어떻게 됩니까.
4. 위치 인코딩를 **곱하기**로 바꾸면 어떻게 됩니까. 왜 더하기입니까.""",

"w_len": """## 정리

- **길이 64에서 LSTM이 무너집니다.** 학습률 3종 어디에서도 0.499입니다.
  어텐션은 1.000을 냅니다.
- **거리와 무관하기 때문입니다.** LSTM은 64걸음을 거쳐 정보를 나르지만,
  어텐션은 0번 자리와 40번 자리를 **한 번에** 견줍니다.
- 그리고 **길어질수록 어텐션이 더 빠릅니다.** 순환은 걸음마다 기다려야
  하지만 어텐션은 전부 동시에 계산합니다.

### ★ 다만 공짜가 아닙니다

어텐션은 모든 자리 쌍을 견주므로 계산량이 길이의 **제곱**에 비례합니다.
길이 64는 괜찮지만 길이 10,000이면 1억 쌍입니다.
**긴 문맥을 다루는 연구의 상당 부분이 이 제곱을 줄이는 일입니다.**

### 연습

1. 길이 128, 256으로 늘리십시오. 어텐션은 언제까지 버팁니까.
2. 길이별 **학습 시간**을 재십시오. LSTM과 어텐션의 시간이 어떻게 벌어집니까.
3. Conv1D의 `kernel_size` 를 길이에 맞춰 키우면 따라옵니까.
   파라미터는 얼마나 늘어납니까.""",

"w_w": """## 정리

- 어텐션 가중치는 **모델이 어디를 봤는지**를 그대로 보여 줍니다.
  9장의 Grad-CAM처럼 따로 계산할 필요가 없습니다. **구조 안에 있습니다.**
- 다만 **가중치가 곧 설명은 아닙니다.** 어디를 봤는지는 알려 주지만
  왜 그렇게 판단했는지는 알려 주지 않습니다. 이 구분은 중요합니다.

### 연습

1. 여러 문장에 대해 그려 보고 **공통된 무늬**가 있는지 보십시오.
2. 위치 인코딩를 뺀 모델의 가중치를 그리십시오. 무엇이 달라집니까.
3. 머리 2개가 **서로 다른 것을 보는지** 확인하십시오.
4. (논술) 어텐션 가중치를 「설명」이라고 부르는 것이 정당한지 논하시오.""",
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
        base = R / "notebooks" / ed / "ch12"
        head = [md(MD["setup"]), code(BOOTSTRAP), code(SETUP),
                md(MD["data"]), code(DATA),
                md(MD["pos"]), code(POS),
                md(MD["train"]), code(trainer)]

        write(base / "ch12_attention.ipynb",
              [md(MD["t_attn"].format(ko=KO[ed]))] + head
              + [md(MD["run"]), code(RUN),
                 md(MD["shuffle"]), code(SHUFFLE),
                 md(MD["w_attn"])],
              {"chapter": 12, "edition": ed, "title": "어텐션과 위치 인코딩"})

        write(base / "ch12_length.ipynb",
              [md(MD["t_len"].format(ko=KO[ed]))] + head
              + [md(MD["length"]), code(LENGTH_EXP), md(MD["w_len"])],
              {"chapter": 12, "edition": ed, "title": "문장이 길어지면"})

        write(base / "ch12_lr_trap.ipynb",
              [md(MD["t_trap"].format(ko=KO[ed]))] + head
              + [md(MD["lr_trap"]), code(LR_TRAP), md(MD["w_trap"])],
              {"chapter": 12, "edition": ed, "title": "학습률 하나로 고정하면"})

        write(base / "ch12_weights.ipynb",
              [md(MD["t_w"].format(ko=KO[ed]))] + head
              + [md(MD["weights"]), code(WEIGHTS),
                 md(MD["w_w"])],
              {"chapter": 12, "edition": ed, "title": "모델이 어디를 보는가"})
    print("완료")
