# -*- coding: utf-8 -*-
"""14장 실습 노트북 3판 생성 — GAN."""
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

DATA = '''LATENT = 32

s = data.mnist()
x_train = s.x_train.reshape(-1, 784) * 2 - 1     # tanh 출력에 맞춰 [-1, 1]
print(f"학습 {x_train.shape}, 범위 [{x_train.min():.0f}, {x_train.max():.0f}]")

def show(rows, titles, n=8):
    fig, axes = plt.subplots(len(rows), n, figsize=(1.15 * n, 1.25 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (imgs, t) in enumerate(zip(rows, titles)):
        for c in range(n):
            axes[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(t, fontsize=9)
    plt.tight_layout(); plt.show()

def diversity(A, n=300, seed=1):
    """생성물끼리의 평균 거리. **모드 붕괴를 재는 가장 간단한 지표.**

    다 비슷하게 생겼으면 이 값이 작아진다.
    """
    idx = np.random.default_rng(seed).choice(len(A), min(n, len(A)), replace=False)
    B = A[idx]
    d = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(B), 1)].mean())

REAL_DIV = diversity(x_train[:2000])
print(f"진짜 숫자끼리의 평균 거리 {REAL_DIV:.3f}  ← 이것이 목표치입니다")
dlbook.record("ch14_real_diversity", REAL_DIV)'''

AE_FAIL = """# 13장의 오토인코더로 「생성」을 해 봅니다. 디코더에 난수를 넣습니다.
enc, dec = train_autoencoder(x_train, LATENT)

Z = encode(enc, x_train[:5000])
print("학습 데이터가 실제로 만든 잠재 벡터의 분포 —")
print(f"  평균 {Z.mean():.3f}   표준편차 {Z.std():.3f}   0인 성분의 비율 {float((Z == 0).mean()):.3f}")
print()
print("★ **표준정규분포가 전혀 아닙니다.** ReLU라 음수가 없고, 평균이 7 넘습니다.")
print("  그런데 '생성'을 하려면 어떤 분포에서 뽑아야 하는지 알아야 합니다.")
print("  **오토인코더는 그것을 알려 주지 않습니다.**")
print()

rnd = np.random.default_rng(0).normal(size=(500, LATENT)).astype("float32")
gen_rand = decode(dec, rnd)
gen_real = decode(dec, Z[:500])

print(f"{'무엇을 디코더에 넣었나':<28}{'생성물끼리 평균 거리':>22}")
print("-" * 52)
print(f"{'표준정규분포 난수':<28}{diversity(gen_rand):>22.3f}")
print(f"{'실제 데이터의 잠재 벡터':<28}{diversity(gen_real):>22.3f}")
print(f"{'(진짜 숫자)':<28}{REAL_DIV:>22.3f}")
dlbook.record("ch14_ae_random_diversity", diversity(gen_rand))
dlbook.record("ch14_ae_real_diversity", diversity(gen_real))

show([gen_real[:8], gen_rand[:8]], ["실제 잠재", "난수"])

print()
print("→ 실제 잠재 벡터를 넣으면 숫자가 나옵니다. **난수를 넣으면 얼룩입니다.**")
print("→ 잠재 공간의 대부분이 **비어 있기** 때문입니다.")
print("→ 생성을 하려면 **어디가 채워져 있는지**를 알아야 합니다. 그것이 14장입니다.")"""

GAN_RUN = """# GAN 학습. 두 모델이 서로를 이기려 합니다.
epochs = 5 if dlbook.smoke.is_smoke() else 30
checkpoints = [1, 2, 5] if dlbook.smoke.is_smoke() else [1, 2, 5, 10, 20, 30]

G, D, hist = train_gan(x_train[:20000], LATENT, epochs=epochs,
                       checkpoints=checkpoints, diversity_fn=diversity)

print()
print(f"{'epoch':<8}{'D 손실':>10}{'G 손실':>10}{'다양성':>10}")
print("-" * 38)
for e, dl, gl, div in hist:
    print(f"{e:<8}{dl:>10.3f}{gl:>10.3f}{div:>10.3f}")
    dlbook.record(f"ch14_div_epoch{e}", div)
    dlbook.record(f"ch14_gloss_epoch{e}", gl)
print(f"{'(진짜)':<8}{'-':>10}{'-':>10}{REAL_DIV:>10.3f}")

print()
print("★ **G 손실이 올라가는데 결과는 좋아집니다.**")
print("  지도학습에서는 손실이 내려가야 좋은 것이었습니다. 여기서는 아닙니다.")
print("  D도 같이 좋아지고 있어서, G의 손실은 **G의 실력이 아니라 둘의 균형**을 잽니다.")
print()
print("★ 다양성이 목표치에 **못 미친 채 멈춥니다.** 부분적인 모드 붕괴입니다.")"""

GAN_SHOW = """z = np.random.default_rng(0).normal(size=(8, LATENT)).astype("float32")
show([x_train[:8], generate(G, z)], ["진짜", "GAN 생성"])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ep = [h[0] for h in hist]
ax[0].plot(ep, [h[1] for h in hist], "o-", label="D 손실")
ax[0].plot(ep, [h[2] for h in hist], "s-", label="G 손실")
ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("손실 — 수렴하지 않습니다")
ax[1].plot(ep, [h[3] for h in hist], "o-", color="#2a9d8f")
ax[1].axhline(REAL_DIV, ls="--", color="#999999")
ax[1].annotate("진짜 데이터", (ep[0], REAL_DIV), xytext=(0, 5),
               textcoords="offset points", fontsize=9, color="#666666")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("생성물끼리 평균 거리")
ax[1].grid(alpha=0.3); ax[1].set_title("다양성 — 목표치에 못 미칩니다")
plt.tight_layout(); plt.show()

print("→ 왼쪽: **손실만 보고는 학습이 잘 되는지 알 수 없습니다.**")
print("→ 오른쪽: 다양성은 늘다가 멈춥니다. 진짜(점선)에 닿지 못합니다.")
print()
print("★ **GAN에는 「검증 손실」에 해당하는 것이 없습니다.**")
print("  7장에서 배운 조기 종료·모델 선택이 그대로 통하지 않습니다.")
print("  그래서 실무에서는 **눈으로 보고** 고르거나 별도 지표(FID 등)를 씁니다.")"""

TRAIN = {
"keras": '''import keras
from keras import layers, ops

def make_generator(latent):
    """잡음 → 그림. **이 함수만 판마다 다릅니다.**"""
    return keras.Sequential([
        layers.Input(shape=(latent,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(784, activation="tanh"),
    ], name="generator")

def make_discriminator():
    """그림 → 진짜일 로짓."""
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="leaky_relu"),
        layers.Dense(128, activation="leaky_relu"),
        layers.Dense(1),
    ], name="discriminator")

class GAN(keras.Model):
    """Keras 3 의 방식 — train_step 하나만 바꿉니다.

    fit() 는 그대로 쓰면서 **한 걸음에 무엇을 할지**만 다시 씁니다.
    백엔드(TensorFlow/PyTorch/JAX)와 무관하게 돕니다.
    """
    def __init__(self, generator, discriminator, latent):
        super().__init__()
        self.G, self.D, self.latent = generator, discriminator, latent
        self.d_loss = keras.metrics.Mean(name="d_loss")
        self.g_loss = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss]

    def compile(self, d_opt, g_opt):
        super().compile()
        self.d_opt, self.g_opt = d_opt, g_opt
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def _d_loss(self, real, fake):
        return (self.bce(ops.ones_like(self.D(real)), self.D(real))
                + self.bce(ops.zeros_like(self.D(fake)), self.D(fake)))

    def train_step(self, real):
        n = ops.shape(real)[0]

        # ① 판별기 — 진짜는 1, 가짜는 0 이라고 답하도록
        z = keras.random.normal((n, self.latent))
        fake = self.G(z)
        dl, grads = self._grad(lambda: self._d_loss(real, fake),
                               self.D.trainable_weights)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_weights))

        # ② 생성기 — 판별기가 **1이라고 답하도록**
        z = keras.random.normal((n, self.latent))
        def gloss():
            out = self.D(self.G(z))
            return self.bce(ops.ones_like(out), out)
        gl, grads = self._grad(gloss, self.G.trainable_weights)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_weights))

        self.d_loss.update_state(dl); self.g_loss.update_state(gl)
        return {"d_loss": self.d_loss.result(), "g_loss": self.g_loss.result()}

    def _grad(self, fn, weights):
        """백엔드에 맞는 방식으로 (손실, 기울기) 를 구한다."""
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf
            with tf.GradientTape() as tape:
                loss = fn()
            return loss, tape.gradient(loss, weights)
        import torch
        loss = fn()
        for w in weights:
            if w.value.grad is not None:
                w.value.grad = None
        loss.backward()
        return loss, [w.value.grad for w in weights]

def train_gan(x, latent, epochs=30, checkpoints=(1, 5, 30),
              lr=2e-4, batch=128, seed=42, diversity_fn=None):
    dlbook.set_seed(seed)
    G, D = make_generator(latent), make_discriminator()
    gan = GAN(G, D, latent)
    gan.compile(d_opt=keras.optimizers.Adam(lr, beta_1=0.5),
                g_opt=keras.optimizers.Adam(lr, beta_1=0.5))
    z_fix = np.random.default_rng(0).normal(size=(500, latent)).astype("float32")
    hist = []
    for e in range(1, epochs + 1):
        h = gan.fit(x, batch_size=batch, epochs=1, verbose=0)
        if e in checkpoints:
            div = diversity_fn(generate(G, z_fix)) if diversity_fn else 0.0
            hist.append((e, float(h.history["d_loss"][-1]),
                         float(h.history["g_loss"][-1]), div))
    return G, D, hist

def generate(G, z):
    return np.asarray(G.predict(z, verbose=0))

def train_autoencoder(x, latent, epochs=15, seed=42):
    dlbook.set_seed(seed)
    enc = keras.Sequential([layers.Input(shape=(784,)),
                            layers.Dense(128, activation="relu"),
                            layers.Dense(latent, activation="relu")])
    dec = keras.Sequential([layers.Input(shape=(latent,)),
                            layers.Dense(128, activation="relu"),
                            layers.Dense(784, activation="tanh")])
    ae = keras.Sequential([enc, dec])
    ae.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    ae.fit(x, x, epochs=dlbook.smoke.epochs(epochs), batch_size=256, verbose=0)
    return enc, dec

def encode(enc, x):
    return np.asarray(enc.predict(x, verbose=0))

def decode(dec, z):
    return np.asarray(dec.predict(z, verbose=0))''',

"tensorflow": '''import tensorflow as tf

L_ = tf.keras.layers

def make_generator(latent):
    return tf.keras.Sequential([
        L_.Input(shape=(latent,)),
        L_.Dense(128, activation="relu"),
        L_.Dense(256, activation="relu"),
        L_.Dense(784, activation="tanh")], name="generator")

def make_discriminator():
    return tf.keras.Sequential([
        L_.Input(shape=(784,)),
        L_.Dense(256, activation="leaky_relu"),
        L_.Dense(128, activation="leaky_relu"),
        L_.Dense(1)], name="discriminator")

def train_gan(x, latent, epochs=30, checkpoints=(1, 5, 30),
              lr=2e-4, batch=128, seed=42, diversity_fn=None):
    """TensorFlow 의 방식 — tf.function 과 GradientTape 를 직접 씁니다."""
    dlbook.set_seed(seed)
    G, D = make_generator(latent), make_discriminator()
    g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def step(real):
        n = tf.shape(real)[0]
        # ① 판별기
        z = tf.random.normal((n, latent))
        with tf.GradientTape() as tape:
            fake = G(z, training=True)
            dl = (bce(tf.ones_like(D(real, training=True)), D(real, training=True))
                  + bce(tf.zeros_like(D(fake, training=True)), D(fake, training=True)))
        d_opt.apply_gradients(zip(tape.gradient(dl, D.trainable_weights),
                                  D.trainable_weights))
        # ② 생성기
        z = tf.random.normal((n, latent))
        with tf.GradientTape() as tape:
            out = D(G(z, training=True), training=True)
            gl = bce(tf.ones_like(out), out)
        g_opt.apply_gradients(zip(tape.gradient(gl, G.trainable_weights),
                                  G.trainable_weights))
        return dl, gl

    ds = (tf.data.Dataset.from_tensor_slices(x).shuffle(4096)
          .batch(batch, drop_remainder=True))
    z_fix = np.random.default_rng(0).normal(size=(500, latent)).astype("float32")
    hist = []
    for e in range(1, epochs + 1):
        dls, gls = [], []
        for real in ds:
            d_, g_ = step(real); dls.append(float(d_)); gls.append(float(g_))
        if e in checkpoints:
            div = diversity_fn(generate(G, z_fix)) if diversity_fn else 0.0
            hist.append((e, float(np.mean(dls)), float(np.mean(gls)), div))
    return G, D, hist

def generate(G, z):
    return np.asarray(G.predict(z, verbose=0))

def train_autoencoder(x, latent, epochs=15, seed=42):
    dlbook.set_seed(seed)
    enc = tf.keras.Sequential([L_.Input(shape=(784,)),
                               L_.Dense(128, activation="relu"),
                               L_.Dense(latent, activation="relu")])
    dec = tf.keras.Sequential([L_.Input(shape=(latent,)),
                               L_.Dense(128, activation="relu"),
                               L_.Dense(784, activation="tanh")])
    ae = tf.keras.Sequential([enc, dec])
    ae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    ae.fit(x, x, epochs=dlbook.smoke.epochs(epochs), batch_size=256, verbose=0)
    return enc, dec

def encode(enc, x):
    return np.asarray(enc.predict(x, verbose=0))

def decode(dec, z):
    return np.asarray(dec.predict(z, verbose=0))''',

"pytorch": '''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_generator(latent):
    return nn.Sequential(nn.Linear(latent, 128), nn.ReLU(),
                         nn.Linear(128, 256), nn.ReLU(),
                         nn.Linear(256, 784), nn.Tanh()).to(device)

def make_discriminator():
    return nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2),
                         nn.Linear(256, 128), nn.LeakyReLU(0.2),
                         nn.Linear(128, 1)).to(device)

def train_gan(x, latent, epochs=30, checkpoints=(1, 5, 30),
              lr=2e-4, batch=128, seed=42, diversity_fn=None):
    """PyTorch 의 방식 — 걸음마다 손으로 씁니다. **가장 훤히 보입니다.**"""
    dlbook.set_seed(seed)
    G, D = make_generator(latent), make_discriminator()
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    dl_ = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                     batch_size=batch, shuffle=True, drop_last=True)
    z_fix = np.random.default_rng(0).normal(size=(500, latent)).astype("float32")
    hist = []
    for e in range(1, epochs + 1):
        dls, gls = [], []
        for (real,) in dl_:
            real = real.to(device); n = real.size(0)
            # ① 판별기 — 진짜는 1, 가짜는 0
            z = torch.randn(n, latent, device=device)
            fake = G(z).detach()               # G로 기울기가 넘어가지 않게
            d_loss = (bce(D(real), torch.ones(n, 1, device=device))
                      + bce(D(fake), torch.zeros(n, 1, device=device)))
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            # ② 생성기 — 판별기가 1이라고 답하도록
            z = torch.randn(n, latent, device=device)
            g_loss = bce(D(G(z)), torch.ones(n, 1, device=device))
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
            dls.append(float(d_loss)); gls.append(float(g_loss))
        if e in checkpoints:
            div = diversity_fn(generate(G, z_fix)) if diversity_fn else 0.0
            hist.append((e, float(np.mean(dls)), float(np.mean(gls)), div))
    return G, D, hist

def generate(G, z):
    G.eval()
    with torch.no_grad():
        out = G(torch.tensor(z, dtype=torch.float32).to(device))
    G.train()
    return out.cpu().numpy()

def train_autoencoder(x, latent, epochs=15, seed=42):
    dlbook.set_seed(seed)
    enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                        nn.Linear(128, latent), nn.ReLU()).to(device)
    dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(),
                        nn.Linear(128, 784), nn.Tanh()).to(device)
    ae = nn.Sequential(enc, dec)
    opt = torch.optim.Adam(ae.parameters(), lr=0.001); loss_fn = nn.MSELoss()
    dl_ = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                     batch_size=256, shuffle=True)
    for _ in range(dlbook.smoke.epochs(epochs)):
        for (xx,) in dl_:
            xx = xx.to(device)
            opt.zero_grad(); loss_fn(ae(xx), xx).backward(); opt.step()
    return enc, dec

def encode(enc, x):
    enc.eval()
    with torch.no_grad():
        out = enc(torch.tensor(x, dtype=torch.float32).to(device))
    return out.cpu().numpy()

def decode(dec, z):
    dec.eval()
    with torch.no_grad():
        out = dec(torch.tensor(z, dtype=torch.float32).to(device))
    return out.cpu().numpy()''',
}

MD = {
"t_ae": """# 14장 실습 ① — 오토인코더로는 왜 생성이 안 되는가

**{ko} 판**

13장의 디코더에 **난수를 넣어** 봅니다. 숫자가 나올까요.""",
"t_gan": """# 14장 실습 ② — GAN

**{ko} 판**

두 모델이 서로를 이기려 합니다. 그 다툼에서 그림이 나옵니다.

> **주의 — 이 노트북의 손실 곡선은 「내려가면 좋은 것」이 아닙니다.**""",
"setup": "## 14.0 준비",
"data": """## 14.1 데이터와 지표

**다양성 지표**를 먼저 정의합니다. 생성물끼리 얼마나 다른지 재는 값입니다.
모드 붕괴를 눈이 아니라 **숫자로** 보기 위해서입니다.""",
"train": """## 14.2 모델과 학습 루프 — 여기만 판마다 다릅니다

**세 판의 차이가 이 장에서 가장 큽니다.**

| 판 | 방식 |
|---|---|
| Keras 3 | `keras.Model` 을 상속해 `train_step` 만 다시 씁니다 |
| TensorFlow | `tf.function` + `GradientTape` 를 직접 씁니다 |
| PyTorch | 걸음마다 손으로 씁니다 — **가장 훤히 보입니다** |

GAN처럼 **한 걸음에 두 모델을 번갈아 갱신**해야 하는 경우,
`fit()` 한 줄로는 안 됩니다.""",
"ae_fail": """## 14.3 오토인코더의 디코더에 난수를 넣으면""",
"gan_run": """## 14.3 GAN 학습""",
"gan_show": """## 14.4 결과와 손실 곡선""",

"w_ae": """## 정리

| 디코더에 넣은 것 | 생성물끼리 평균 거리 |
|---|:--:|
| 표준정규분포 난수 | **11.16** |
| 실제 데이터의 잠재 벡터 | 18.59 |
| (진짜 숫자) | 20.29 |

- **실제 잠재 벡터를 넣으면 숫자가 나오고, 난수를 넣으면 얼룩이 나옵니다.**
- 학습된 잠재 벡터의 분포는 평균 7.5, 표준편차 6.8이고 **28%가 0**입니다.
  표준정규분포와 전혀 다릅니다.
- **오토인코더는 「어디가 채워져 있는지」를 알려 주지 않습니다.**
  그래서 생성 모델이 아닙니다.

> 이 문제를 푸는 길이 둘 있습니다.
> **① 잠재 분포를 알게 만든다** → VAE (15장 §15.5)
> **② 분포를 몰라도 되게 만든다** → **GAN (다음 노트북)**

### 연습

1. 잠재 벡터의 실제 분포에서 뽑아 보십시오(각 차원의 평균·표준편차 사용).
   난수보다 나아집니까.
2. 인코더의 마지막 활성화를 `relu` 대신 없애면 분포가 어떻게 됩니까.
3. 두 숫자의 잠재 벡터를 **직선으로 이어** 중간 점들을 디코딩하십시오.
   중간에 숫자가 아닌 것이 나옵니까.""",

"w_gan": """## 정리

| epoch | D 손실 | G 손실 | 다양성 |
|:--:|:--:|:--:|:--:|
| 1 | 0.736 | 0.968 | 4.84 |
| 5 | 0.757 | 1.806 | 7.98 |
| 10 | 0.834 | 1.597 | 12.41 |
| 20 | 0.723 | 2.182 | 15.73 |
| 30 | 0.582 | 2.216 | 15.64 |
| (진짜) | — | — | **20.29** |

- **★ G 손실이 올라가는데 결과는 좋아집니다.** 지도학습의 직관이 통하지
  않습니다. G의 손실은 **G의 실력이 아니라 둘의 균형**을 잽니다.
- **손실은 수렴하지 않습니다.** 오르내립니다. 그것이 정상입니다.
- **다양성이 목표치에 못 미친 채 멈춥니다** (15.6 대 20.3).
  부분적인 **모드 붕괴**입니다.
- **GAN에는 「검증 손실」에 해당하는 것이 없습니다.** 7장에서 배운
  조기 종료와 모델 선택이 그대로 통하지 않습니다.

### 연습

1. epoch을 60, 100으로 늘리십시오. 다양성이 20.3에 닿습니까.
2. D를 **한 걸음에 두 번** 갱신하십시오. 무엇이 달라집니까.
3. 학습률을 0.001로 올리십시오. 며칠 안에 무너집니까.
4. 생성기의 마지막 활성화를 `tanh` 에서 `sigmoid` 로 바꾸면 어떻게 됩니까.
   (힌트: 데이터의 범위가 [-1, 1] 입니다)
5. **[열린 문제]** 다양성 지표를 클래스별로 재십시오 — 분류기를 하나
   따로 학습시켜 생성물의 숫자를 판정하고, **어떤 숫자가 적게 나오는지**
   보십시오. 그것이 모드 붕괴의 정확한 모습입니다.""",
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
        base = R / "notebooks" / ed / "ch14"
        head = [md(MD["setup"]), code(BOOTSTRAP), code(SETUP),
                md(MD["data"]), code(DATA),
                md(MD["train"]), code(trainer)]

        write(base / "ch14_why_not_ae.ipynb",
              [md(MD["t_ae"].format(ko=KO[ed]))] + head
              + [md(MD["ae_fail"]), code(AE_FAIL), md(MD["w_ae"])],
              {"chapter": 14, "edition": ed, "title": "오토인코더로는 왜 안 되는가"})

        write(base / "ch14_gan.ipynb",
              [md(MD["t_gan"].format(ko=KO[ed]))] + head
              + [md(MD["gan_run"]), code(GAN_RUN),
                 md(MD["gan_show"]), code(GAN_SHOW),
                 md(MD["w_gan"])],
              {"chapter": 14, "edition": ed, "title": "GAN"})
    print("완료")
