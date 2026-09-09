# 한 학기 딥러닝 교과서 — 실습 저장소

> 설명이 먼저, 수식은 그 다음 — 비전공자도 따라오고 전공자도 남는 것이 있는 15주
> 케라스 · 텐서플로 · 파이토치 세 판 · 설치 없이 실습

이 저장소는 **「한 학기 딥러닝 교과서」의 실습 코드 전체**입니다.
본문에 나오는 **모든 수치가 여기서 실제로 실행된 값**이고,
CI가 원고와 대조합니다.

---

## 무엇이 들어 있나

| | |
|---|---|
| 노트북 | **3판 × 35개 = 105개** (Keras 3 / TensorFlow / PyTorch) |
| 검증 수치 | **466개** (`expected.json`, 판별로 구분) |
| 그림 | SVG 45장 (`figures/`) |
| 공통 유틸 | `dlbook/` — **numpy만 씁니다. 프레임워크에 무관합니다** |

---

## 세 판

같은 실험을 세 가지 방식으로 냅니다.

| 판 | 성격 | 학습 코드 |
|---|---|---|
| **Keras 3** | *개념* | `Sequential` + `fit()` |
| **TensorFlow** | *TF의 방식* | `tf.data` + `GradientTape` |
| **PyTorch** | *PyTorch의 방식* | `nn.Module` + `DataLoader` + 수동 루프 |

::: 중요
**세 판이 다른 것은 「모델 정의」와 「학습 루프」 두 셀뿐입니다.**
나머지는 글자 하나까지 같습니다. 노트북을 나란히 열면 차이가
곧바로 보입니다.
:::

강의자는 **한 판을 고르시면 됩니다.** 부록 A에 대조표가 있습니다.

---

## 빠른 시작

### 브라우저에서 (설치 없음)

Kaggle Notebooks 또는 Colab에서 노트북을 열고 첫 셀을 실행하십시오.

```python
try:
    import dlbook
except ImportError:
    !pip install -q "dlbook @ git+https://github.com/<계정>/deep-learning-in-one-semester.git"
    import dlbook
```

### 로컬에서

```bash
git clone https://github.com/<계정>/deep-learning-in-one-semester.git
cd deep-learning-in-one-semester

pip install -r requirements-keras.txt     # 또는 -tensorflow / -pytorch
pip install -e .

jupyter lab notebooks/keras/
```

---

## 장별 노트북

| 장 | 노트북 | 무엇을 보이는가 |
|:--:|---|---|
| 2 | `ch02_delta`, `ch02_perceptron`, `ch02_xor` | 델타 규칙 → 프레임워크, XOR |
| 3 | `ch03_three_frameworks`, `ch03_backend_swap` | 세 판 대조, Keras 백엔드 전환 |
| 5 | `ch05_learning_rate`, `ch05_depth`, `ch05_init` | **깊이의 벽은 학습률이었다** |
| 6 | `ch06_overfitting`, `ch06_regularization`, `ch06_batchnorm` | 규제, 배치 정규화의 진짜 일 |
| 7 | `ch07_metrics`, `ch07_selection_bias` | **선택 편향은 「몇 번 골랐나」** |
| 8 | `ch08_dnn_vs_cnn`, `ch08_no_activation`, `ch08_mnist` | **활성화를 빼도 최대 풀링이 비선형** |
| 9 | `ch09_transfer`, `ch09_pretrained`, `ch09_gradcam` | 전이학습, Grad-CAM |
| 10 | `ch10_memory`, `ch10_lr`, `ch10_seed`, `ch10_forecast` | **★ 시드만 바꿔도 0.46↔1.00** |
| 11 | `ch11_representation`, `ch11_embedding_space` | **임베딩만으로는 순서를 못 본다** |
| 12 | `ch12_attention`, `ch12_length`, `ch12_lr_trap`, `ch12_weights` | **어텐션도 순서를 못 본다 → 위치 부호화** |
| 13 | `ch13_autoencoder`, `ch13_denoise`, `ch13_anomaly`, `ch13_anomaly_ood` | **선형 AE = PCA (넷째 자리까지)** |
| 14 | `ch14_why_not_ae`, `ch14_gan` | **손실이 오르는데 결과는 좋아진다** |

---

## 이 저장소의 규율

### ① 본문의 숫자를 손으로 옮겨 적지 않습니다

```python
acc = metrics.accuracy(y_test, pred)
dlbook.record("ch11_lstm_acc", acc)      # → expected.json 과 대조
```

**이 규율이 실제로 초고의 오류를 네 번 잡았습니다.**

| 장 | 초고에 쓴 것 | 실측 |
|:--:|---|---|
| 5 | "ReLU도 20층에서 무너진다" | 학습률 문제였습니다 |
| 7 | "test를 val로 쓰면 크게 부풀려진다" | 거의 안 나왔습니다 |
| 8 | "활성화를 빼면 선형이다" | 최대 풀링이 비선형입니다 |
| 12 | "LSTM은 길이 32에서 못 한다" | lr=0.001에서는 1.000 |

### ② 노트북 하나는 10분을 넘지 않습니다

실습 60분 안에 **돌리고 → 바꿔서 다시 돌리고 → 견주기**가
들어가야 하기 때문입니다. `run_notebooks.py` 가 600초를 넘으면
경고합니다.

### ③ 플랫폼 중립 8규칙

Colab 전용 코드, 절대 경로, 중간 `pip install` 을 금지합니다.
`lint_notebooks.py` 가 검사합니다.

### ④ 세 판의 절 구성이 같아야 합니다

`check_parity.py` 가 검사합니다.

### ⑤ 인터넷 없이도 대부분 돌아갑니다

2·5·6·7·10·11·12장은 **합성 데이터**로 전부 돌아갑니다. 내려받는 것이 없습니다.
8·9·13·14장은 표준 데이터셋이 필요한데, **코드가 알아서 받습니다.**
막힌 곳에서 쓸 것이라면 강의 전날 한 번 미리 받아 두십시오.

```bash
python -m dlbook.fetch --dest data/ --weights
```

(부록 C 참조)

---

## 도구

```bash
# 노트북 검사
python tools/lint_notebooks.py notebooks/

# 세 판 절 구성 대조
python tools/check_parity.py

# 실행 + 수치 대조
python tools/run_notebooks.py notebooks/keras --check-expected

# CI용 축소 실행 (epoch 1, 데이터 1/20)
python tools/run_notebooks.py notebooks/keras --smoke

# 노트북 다시 만들기 (직접 고치지 마십시오)
python tools/gen/gen_ch12.py

# 그림 다시 만들기
python figures/make_figures.py
```

---

## 데이터

`dlbook.data` 는 **로컬을 먼저 뒤진 뒤** 없으면 내려받습니다.

| 순서 | 위치 |
|:--:|---|
| 1 | `$DLBOOK_DATA` |
| 2 | `<저장소>/data/` |
| 3 | `./data/` |
| 4 | `~/.dlbook/datasets/` |

**데이터 파일은 git에 올리지 않습니다.** 전부 코드가 내려받습니다.
미리 받아 두려면:

```bash
python -m dlbook.fetch                  # ~/.dlbook/datasets 에
python -m dlbook.fetch --dest data/     # 저장소의 data/ 에
python -m dlbook.fetch --weights        # ImageNet 사전학습 가중치까지
```

받다가 끊겨도 **반쯤 받은 파일이 남지 않습니다.** `.part` 로 받아서 다 받은
뒤에 제자리로 옮깁니다.

사전학습 가중치(9장)는 `data/pretrained/` 가 있으면 케라스와 파이토치가
그곳을 보도록 `dlbook` 이 `KERAS_HOME` · `TORCH_HOME` 을 맞춰 줍니다.
그 폴더만 복사해 가면 인터넷 없이 전이학습 실습이 돕니다.

**`mnist()`, `fashion_mnist()`, `cifar10()` 은 keras를 거치지
않습니다.** 원본 파일을 직접 읽습니다 — PyTorch만 쓰는 노트북에
TensorFlow가 딸려 오면 브라우저 환경에서 커널이 죽기 때문입니다.

---

## 라이선스

코드: MIT. 본문과 그림은 별도입니다.
