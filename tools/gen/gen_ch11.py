# -*- coding: utf-8 -*-
"""11장 실습 노트북 3판 생성 — 텍스트와 임베딩."""
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

DATA = """# 어순이 정답을 바꾸는 감성 분류. 인터넷 없이 만듭니다. (본문 §11.4)
V = len(data.TOY_VOCAB)
x, y = data.toy_reviews(n=8000, length=16, seed=42)
sp = data.split(x, y, val_ratio=0.2, test_ratio=0.2, seed=42)
print(sp.summary())
print(f"어휘 {V}개:", " ".join(data.TOY_VOCAB))
print()

for k in range(4):
    print(f"  {data.toy_decode(x[k]):<48} → {'긍정' if y[k] else '부정'}")

print()
print("★ 규칙 — 문장에 감성 단어 하나와 미끼 단어 하나가 있고,")
print("  부정어 '안'이 **둘 중 하나 앞에** 붙습니다.")
print("  '안'이 감성 단어에 붙었으면 정답이 뒤집힙니다.")
print()
print("  영화 안 좋다 정말 배우   → 부정")
print("  영화 좋다 안 정말 배우   → 긍정")
print()
print("  **두 문장에 든 단어의 집합은 같습니다.** 다른 것은 순서뿐입니다.")
print("  그래서 단어 가방으로는 **원리적으로** 풀 수 없습니다.")


def to_bag(seqs, vocab_size=V):
    \"\"\"단어 가방 — 각 단어가 몇 번 나왔는지만 센다. **순서를 버린다.**\"\"\"
    b = np.stack([np.bincount(r, minlength=vocab_size) for r in seqs])
    b[:, 0] = 0                      # <pad>는 세지 않는다
    return b.astype("float32")"""

RUN = """kinds = [("bow",  "단어 가방 + Dense"),
         ("avg",  "임베딩 + 평균"),
         ("cnn",  "임베딩 + Conv1D"),
         ("lstm", "임베딩 + LSTM")]

base_acc = {}
print(f"{'모델':<22}{'파라미터':>12}{'시험 정확도':>14}")
print("-" * 48)
for kind, ko in kinds:
    acc, n_params, _ = train_text(kind, sp)
    base_acc[kind] = acc
    print(f"{ko:<22}{n_params:>12,}{acc:>14.3f}")
    dlbook.record(f"ch11_{kind}_acc", acc)
    dlbook.record(f"ch11_{kind}_params", n_params)

print()
print("→ 앞의 둘은 **동전 던지기**입니다. 어순을 버렸으니 풀릴 수 없습니다.")
print("→ 순서를 보는 층을 올리는 순간 풀립니다.")
print("   데이터도, 임베딩도, 어휘도 그대로입니다. **읽는 방식만 바꿨습니다.**")
print("→ **파라미터가 성능을 만들지 않습니다.** Conv1D는 단어 가방보다 크지만")
print("   그 차이는 172개뿐이고, 정확도는 0.51에서 1.00으로 갑니다.")"""

SHUFFLE = """# 확인 — 평균이 정말 순서를 지우는가. 시험 문장의 단어를 섞어서 넣습니다.
rng = np.random.default_rng(0)
x_shuf = np.stack([rng.permutation(r) for r in sp.x_test])

print(f"{'모델':<8}{'원래 순서':>12}{'섞은 뒤':>12}{'떨어진 폭':>12}")
print("-" * 44)
for kind in ("avg", "cnn", "lstm"):
    train_text(kind, sp)
    a_ord = predict_acc(kind, sp, sp.x_test)
    a_shuf = predict_acc(kind, sp, x_shuf)
    print(f"{kind:<8}{a_ord:>12.3f}{a_shuf:>12.3f}{a_ord - a_shuf:>12.3f}")
    dlbook.record(f"ch11_shuffle_drop_{kind}", float(a_ord - a_shuf))

print()
print("→ 평균 모델은 **섞어도 그대로**입니다. 애초에 순서를 안 봤기 때문입니다.")
print("→ Conv1D와 LSTM은 무너집니다. **순서를 보고 있었다는 증거**입니다.")"""

HARD = """# 심화 — 규칙을 더 어렵게 바꾸면 Conv1D와 LSTM이 갈립니다.
# hard=True: 감성 단어가 **둘**(부호 반대) 들어가고, 정답은 **먼저 나온 쪽**을 따릅니다.
xh, yh = data.toy_reviews(n=8000, length=16, seed=42, hard=True)
sh = data.split(xh, yh, val_ratio=0.2, test_ratio=0.2, seed=42)

for k in range(3):
    print(f"  {data.toy_decode(xh[k]):<52} → {'긍정' if yh[k] else '부정'}")
print()

print(f"{'모델':<8}{'기본 규칙':>12}{'어려운 규칙':>14}")
print("-" * 36)
for kind, base in [("cnn", "cnn"), ("lstm", "lstm")]:
    acc_h, _, _ = train_text(kind, sh)
    print(f"{kind:<8}{base_acc[base]:>12.3f}{acc_h:>14.3f}")
    dlbook.record(f"ch11_hard_{kind}_acc", acc_h)

print()
print("→ **Conv1D가 처음으로 밀립니다.** 창 크기 3으로는 '안 좋다' 같은")
print("   **붙어 있는** 조합만 봅니다. 두 감성 단어 사이에 채움말이 여럿")
print("   끼어 있으면 어느 쪽이 먼저인지 한 창에 담기지 않습니다.")
print("→ LSTM은 처음부터 끝까지 읽으므로 거리와 무관합니다.")

"""

EMB_TRAIN = """# 학습된 LSTM에서 임베딩 표를 꺼냅니다.
acc, n_params, E = train_text("lstm", sp)
print(f"시험 정확도 {acc:.3f}, 임베딩 표 크기 {E.shape}")
print()
print("이 표는 **학습되는 파라미터**였습니다. 처음에는 난수였습니다.")
print("무엇이 되었는지 열어 봅니다.")"""

EMB_DIST = """# 거리는 **코사인 거리**로 잽니다.
# 프레임워크마다 임베딩 초기값의 크기가 달라(예: 균등 ±0.05 대 정규분포)
# 유클리드 거리는 판마다 자릿수가 달라집니다. 코사인은 방향만 보므로
# **세 판에서 같은 값**이 나옵니다. 0에 가까우면 같은 방향, 1이면 직각,
# 2면 정반대입니다.
def dist(a, b):
    i, j = data.TOY_VOCAB.index(a), data.TOY_VOCAB.index(b)
    u, v = E[i], E[j]
    return float(1.0 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

pairs = [("좋다", "훌륭하다"), ("나쁘다", "지루하다"), ("좋다", "나쁘다"),
         ("좋다", "영화"), ("안", "좋다")]

print(f"{'단어 쌍':<26}{'거리':>8}")
print("-" * 34)
for a, b in pairs:
    d = dist(a, b)
    print(f"{a + ' ↔ ' + b:<26}{d:>8.3f}")
    dlbook.record(f"ch11_dist_{a}_{b}", d)

# 한 쌍이 아니라 **무리끼리** 재 봅니다. 우연이 아님을 보이려면 이쪽이 낫습니다.
POS, NEG = data.TOY_VOCAB[2:7], data.TOY_VOCAB[7:12]
FILLER = data.TOY_VOCAB[12:]

def mean_dist(A, B):
    return float(np.mean([dist(a, b) for a in A for b in B if a != b]))

rows = [("긍정 무리 안에서", mean_dist(POS, POS)),
        ("부정 무리 안에서", mean_dist(NEG, NEG)),
        ("채움 무리 안에서", mean_dist(FILLER, FILLER)),
        ("긍정 ↔ 부정", mean_dist(POS, NEG)),
        ("'안' ↔ 나머지 전부", mean_dist(["안"], POS + NEG + FILLER))]

print()
print(f"{'무리':<26}{'평균 거리':>10}")
print("-" * 36)
for ko, v in rows:
    print(f"{ko:<26}{v:>10.3f}")
dlbook.record("ch11_grp_pos_in", rows[0][1])
dlbook.record("ch11_grp_neg_in", rows[1][1])
dlbook.record("ch11_grp_pos_neg", rows[3][1])
dlbook.record("ch11_grp_neg_word", rows[4][1])

print()
print("★ 데이터에는 '좋다'와 '훌륭하다'가 비슷하다는 정보가 **없었습니다.**")
print("  단어는 번호였고, 정답은 0 아니면 1이었습니다.")
print("  그런데도 같은 역할을 하는 단어끼리 열 배 넘게 가까워졌습니다.")"""

EMB_PCA = """# 8차원을 주성분 2개로 눌러서 그립니다. (numpy만 씁니다)
def pca2(M):
    C = M - M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return C @ Vt[:2].T

P = pca2(E)
groups = {"긍정": data.TOY_VOCAB[2:7], "부정": data.TOY_VOCAB[7:12],
          "채움": data.TOY_VOCAB[12:], "부정어": ["안"]}
colors = {"긍정": "#1f77b4", "부정": "#d62728", "채움": "#999999",
          "부정어": "#ff7f0e"}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for g, words in groups.items():
    idx = [data.TOY_VOCAB.index(w) for w in words]
    ax.scatter(P[idx, 0], P[idx, 1], s=90, c=colors[g], label=g, zorder=3)
    for i in idx:
        ax.annotate(data.TOY_VOCAB[i], (P[i, 0], P[i, 1]),
                    fontsize=9, xytext=(5, 4), textcoords="offset points")
ax.axhline(0, color="#dddddd", lw=0.8); ax.axvline(0, color="#dddddd", lw=0.8)
ax.set_xlabel("주성분 1"); ax.set_ylabel("주성분 2")
ax.set_title("학습된 임베딩 — 아무도 알려 주지 않았습니다")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

print("→ 긍정과 부정이 서로 반대편에 모였습니다.")
print("→ 뜻이 없는 채움 단어들은 가운데에 뭉쳤습니다.")
print("→ **'안'은 어느 무리에도 속하지 않습니다.** 긍정도 부정도 아니고,")
print("   뒤에 오는 단어의 부호를 뒤집는 **연산자**이기 때문입니다.")"""

TRAIN = {
"keras": '''import keras
from keras import layers

def _build(kind, V, L):
    """모델 정의 — **이 함수만 판마다 다릅니다.**"""
    if kind == "bow":
        return keras.Sequential([
            layers.Input(shape=(V,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")])
    ls = [layers.Input(shape=(L,)), layers.Embedding(V, 8)]
    if kind == "avg":
        ls += [layers.GlobalAveragePooling1D(),
               layers.Dense(16, activation="relu")]
    elif kind == "cnn":
        ls += [layers.Conv1D(16, 3, activation="relu"),
               layers.GlobalMaxPooling1D()]
    elif kind == "lstm":
        ls += [layers.LSTM(32)]
    ls += [layers.Dense(1, activation="sigmoid")]
    return keras.Sequential(ls)

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 임베딩 표) 를 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    prep = (lambda z: to_bag(z, V)) if kind == "bow" else (lambda z: z)
    m = _build(kind, V, L)
    m.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
              loss="binary_crossentropy")
    m.fit(prep(sp.x_train), sp.y_train,
          validation_data=(prep(sp.x_val), sp.y_val),
          epochs=dlbook.smoke.epochs(epochs), batch_size=64, verbose=0)
    _FIT[kind] = (m, prep)
    pred = (m.predict(prep(sp.x_test), verbose=0).reshape(-1) > 0.5).astype(int)
    emb = None
    for lyr in m.layers:
        if isinstance(lyr, layers.Embedding):
            emb = lyr.get_weights()[0]
    return metrics.accuracy(sp.y_test, pred), m.count_params(), emb

def predict_acc(kind, sp, xs):
    """이미 학습된 모델로 임의의 입력에 대한 정확도를 잰다."""
    m, prep = _FIT[kind]
    pred = (m.predict(prep(xs), verbose=0).reshape(-1) > 0.5).astype(int)
    return metrics.accuracy(sp.y_test, pred)''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def _build(kind, V, L):
    """모델 정의 — **이 함수만 판마다 다릅니다.**"""
    if kind == "bow":
        return tf.keras.Sequential([
            L_.Input(shape=(V,)),
            L_.Dense(16, activation="relu"),
            L_.Dense(1, activation="sigmoid")])
    ls = [L_.Input(shape=(L,)), L_.Embedding(V, 8)]
    if kind == "avg":
        ls += [L_.GlobalAveragePooling1D(), L_.Dense(16, activation="relu")]
    elif kind == "cnn":
        ls += [L_.Conv1D(16, 3, activation="relu"), L_.GlobalMaxPooling1D()]
    elif kind == "lstm":
        ls += [L_.LSTM(32)]
    ls += [L_.Dense(1, activation="sigmoid")]
    return tf.keras.Sequential(ls)

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 임베딩 표) 를 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    prep = (lambda z: to_bag(z, V)) if kind == "bow" else (lambda z: z)
    m = _build(kind, V, L)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
              loss="binary_crossentropy")
    m.fit(prep(sp.x_train), sp.y_train,
          validation_data=(prep(sp.x_val), sp.y_val),
          epochs=dlbook.smoke.epochs(epochs), batch_size=64, verbose=0)
    _FIT[kind] = (m, prep)
    pred = (m.predict(prep(sp.x_test), verbose=0).reshape(-1) > 0.5).astype(int)
    emb = None
    for lyr in m.layers:
        if isinstance(lyr, L_.Embedding):
            emb = lyr.get_weights()[0]
    return metrics.accuracy(sp.y_test, pred), m.count_params(), emb

def predict_acc(kind, sp, xs):
    """이미 학습된 모델로 임의의 입력에 대한 정확도를 잰다."""
    m, prep = _FIT[kind]
    pred = (m.predict(prep(xs), verbose=0).reshape(-1) > 0.5).astype(int)
    return metrics.accuracy(sp.y_test, pred)''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextModel(nn.Module):
    """임베딩 위에 무엇을 올리느냐만 다른 네 모델.

    Conv1d는 (배치, 채널, 길이) 를 받으므로 **축을 바꿔 줘야** 합니다.
    Keras의 Conv1D는 (배치, 길이, 채널) 을 받습니다. 여기가 다릅니다.
    """
    def __init__(self, kind, V, L, d=8):
        super().__init__()
        self.kind = kind
        if kind == "bow":
            self.body = nn.Sequential(nn.Linear(V, 16), nn.ReLU())
            n_out = 16
        else:
            self.emb = nn.Embedding(V, d)
            if kind == "avg":
                self.body = nn.Sequential(nn.Linear(d, 16), nn.ReLU())
                n_out = 16
            elif kind == "cnn":
                self.conv = nn.Conv1d(d, 16, 3); n_out = 16
            elif kind == "lstm":
                self.rnn = nn.LSTM(d, 32, batch_first=True); n_out = 32
        self.head = nn.Linear(n_out, 1)

    def forward(self, x):
        if self.kind == "bow":
            return self.head(self.body(x))
        e = self.emb(x.long())                      # (B, L, d)
        if self.kind == "avg":
            h = self.body(e.mean(1))
        elif self.kind == "cnn":
            h = torch.relu(self.conv(e.transpose(1, 2))).max(dim=2).values
        else:
            out, _ = self.rnn(e); h = out[:, -1]
        return self.head(h)

_FIT = {}

def train_text(kind, sp, lr=0.003, seed=42, epochs=25):
    """(시험 정확도, 파라미터 수, 임베딩 표) 를 돌려준다."""
    dlbook.set_seed(seed)
    V, L = len(data.TOY_VOCAB), sp.x_train.shape[1]
    prep = (lambda z: to_bag(z, V)) if kind == "bow" else (lambda z: z)
    model = TextModel(kind, V, L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    clip = 1.0                        # 10장 §10.5의 권고 — 기울기 절단
    dl = DataLoader(TensorDataset(
        torch.tensor(prep(sp.x_train), dtype=torch.float32),
        torch.tensor(sp.y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=64, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        model.train()
        for xx, yy in dl:
            xx, yy = xx.to(device), yy.to(device)
            opt.zero_grad(); loss_fn(model(xx), yy).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    _FIT[kind] = (model, prep)
    n_params = sum(p.numel() for p in model.parameters())
    emb = (model.emb.weight.detach().cpu().numpy()
           if kind != "bow" else None)
    return predict_acc(kind, sp, sp.x_test), n_params, emb

def predict_acc(kind, sp, xs):
    """이미 학습된 모델로 임의의 입력에 대한 정확도를 잰다."""
    model, prep = _FIT[kind]
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(prep(xs), dtype=torch.float32).to(device))
    pred = (out.cpu().numpy().reshape(-1) > 0).astype(int)
    return metrics.accuracy(sp.y_test, pred)''',
}

MD = {
"t_rep": """# 11장 실습 ① — 무엇으로 표현하고, 어떻게 읽는가

**{ko} 판**

텍스트를 네 가지 방식으로 표현하고 성능을 견줍니다.

| 모델 | 무엇으로 표현하는가 | 순서를 보는가 |
|---|---|:--:|
| 단어 가방 + Dense | 등장 횟수 | ✗ |
| 임베딩 + 평균 | 학습된 벡터 | ✗ |
| 임베딩 + Conv1D | 학습된 벡터 | 이웃 3개 |
| 임베딩 + LSTM | 학습된 벡터 | 처음부터 끝까지 |

**앞의 둘과 뒤의 둘 사이에서 성능이 갈립니다.** 그 이유가 이 장입니다.""",

"t_emb": """# 11장 실습 ② — 임베딩은 무엇을 배웠나

**{ko} 판**

학습이 끝난 모델에서 **임베딩 표를 꺼내 봅니다.**

데이터에는 *"좋다와 훌륭하다가 비슷한 뜻"* 이라는 정보가 **없었습니다.**
단어는 번호였고 정답은 0 아니면 1이었습니다. 그런데도 —""",

"setup": "## 11.0 준비",
"data": """## 11.1 실험대 — 어순이 정답을 바꾸는 감성 분류

IMDB나 네이버 영화평은 **단어 가방만으로도 0.85 이상**이 나옵니다.
그러면 *"순서를 봐야 한다"* 는 이 장의 논증이 숫자로 드러나지 않습니다.

그래서 **어순이 정답을 바꾸도록** 만든 데이터를 씁니다.
인터넷 없이 만들어집니다. (3장 §3.2)""",
"train": """## 11.2 모델 정의 — 여기만 판마다 다릅니다

**PyTorch 판에서 `Conv1d` 앞에 `transpose(1, 2)` 가 있는 것**에
주목하십시오. PyTorch는 (배치, **채널**, 길이) 를 받고 Keras는
(배치, 길이, **채널**) 을 받습니다. 자주 틀리는 자리입니다.""",
"run": """## 11.3 네 모델을 나란히""",
"shuffle": """## 11.4 확인 — 평균은 정말 순서를 지우는가

§11.7의 주장을 직접 확인합니다. **시험 문장의 단어를 무작위로 섞어서**
넣어 봅니다. 순서를 안 보는 모델이라면 **성능이 그대로여야** 합니다.""",
"hard": """## 11.5 심화 — 규칙을 어렵게 하면 Conv1D가 밀립니다

여기까지는 Conv1D와 LSTM이 똑같이 1.000이었습니다.
**과제가 「붙어 있는 두 단어」로 풀렸기 때문입니다.**

규칙을 바꿔 **먼 거리의 순서 관계**를 묻게 하면 둘이 갈립니다.""",
"emb_train": """## 11.1 학습하고, 임베딩 표를 꺼냅니다""",
"emb_dist": """## 11.2 단어 사이의 거리""",
"emb_pca": """## 11.3 그림으로 — 8차원을 2차원으로""",

"w_rep": """## 정리

| 모델 | 순서를 보는가 | 결과 |
|---|:--:|---|
| 단어 가방 + Dense | ✗ | 동전 던지기 |
| 임베딩 + 평균 | ✗ | 동전 던지기 |
| 임베딩 + Conv1D | 이웃 3개 | 풀림 |
| 임베딩 + LSTM | 전체 | 완전히 풀림 |

- **임베딩은 「무엇으로 표현할지」만 해결합니다.** 순서를 읽으려면
  그 위에 층이 더 필요합니다.
- **평균은 덧셈이고, 덧셈은 순서를 따지지 않습니다.**
  임베딩을 통과한 뒤에 다시 단어 가방이 된 것입니다.
- **파라미터 수가 성능을 만들지 않습니다.** 가장 못한 모델이 가장 큽니다.

### 연습

1. `Conv1D` 의 `kernel_size` 를 3 → 5 → 7로 늘리십시오.
   LSTM에 얼마나 가까워집니까. **왜 그렇습니까.**
2. `toy_reviews(length=32)` 로 문장을 길게 만들고 다시 돌리십시오.
   어느 모델이 가장 많이 떨어집니까.
3. `GlobalAveragePooling1D` 를 `Flatten` 으로 바꾸면 어떻게 됩니까.
   파라미터 수는 어떻게 됩니까.
4. **[열린 문제]** 네이버 영화평(NSMC)이나 IMDB로 같은 표를 만드십시오.
   **단어 가방이 0.5보다 훨씬 높게 나올 것입니다.** 왜입니까.""",

"w_emb": """## 정리

- **임베딩은 학습되는 조회표입니다.** 번호를 받아 벡터를 돌려줄 뿐이고,
  그 표가 역전파로 갱신됩니다.
- 아무도 알려 주지 않았는데 **같은 역할을 하는 단어가 같은 곳으로**
  모였습니다. 모델 입장에서 그것이 손실을 줄이는 가장 쉬운 길이기
  때문입니다.
- 담기는 것은 **뜻**이 아니라 **그 과제에서의 역할**입니다.
  품사 분류로 학습하면 "좋다"와 "나쁘다"가 붙습니다. 둘 다 형용사이기
  때문입니다. **임베딩은 과제의 거울입니다.**

### 연습

1. 임베딩 길이를 8에서 **2**로 줄이십시오. 성능이 유지됩니까.
   길이가 2이므로 **주성분 분석 없이 그대로 그릴 수 있습니다.**
2. Conv1D 모델의 임베딩을 꺼내 같은 그림을 그리십시오.
   LSTM의 것과 배치가 비슷합니까.
3. 학습을 **1 epoch만** 하고 그려 보십시오. 난수에서 얼마나 움직였습니까.
4. `toy_reviews()` 에서 부정어를 없앤 데이터를 만들고(감성 단어만 둘),
   "안"의 벡터가 어떻게 되는지 보십시오. **왜 그렇습니까.**
   (본문 「수식으로 한 번 더」의 역전파 식)""",
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
        base = R / "notebooks" / ed / "ch11"
        head = [md(MD["setup"]), code(BOOTSTRAP), code(SETUP),
                md(MD["data"]), code(DATA),
                md(MD["train"]), code(trainer)]

        write(base / "ch11_representation.ipynb",
              [md(MD["t_rep"].format(ko=KO[ed]))] + head
              + [md(MD["run"]), code(RUN),
                 md(MD["shuffle"]), code(SHUFFLE),
                 md(MD["hard"]), code(HARD),
                 md(MD["w_rep"])],
              {"chapter": 11, "edition": ed, "title": "무엇으로 표현하고 어떻게 읽는가"})

        write(base / "ch11_embedding_space.ipynb",
              [md(MD["t_emb"].format(ko=KO[ed]))] + head
              + [md(MD["emb_train"]), code(EMB_TRAIN),
                 md(MD["emb_dist"]), code(EMB_DIST),
                 md(MD["emb_pca"]), code(EMB_PCA),
                 md(MD["w_emb"])],
              {"chapter": 11, "edition": ed, "title": "임베딩은 무엇을 배웠나"})
    print("완료")
