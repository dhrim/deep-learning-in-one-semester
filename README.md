# 한 학기 딥러닝 교과서 — 실습 저장소

> 설명이 먼저, 수식은 그 다음 — 비전공자도 따라오고 전공자도 남는 것이 있는 15주
> 케라스 · 텐서플로 · 파이토치 세 판 · 설치 없이 실습

이 저장소는 **「한 학기 딥러닝 교과서」의 실습 코드 전체**입니다.

**노트북에 실행 결과가 담겨 있습니다.** 아무것도 설치하지 않고, 내려받지도 않고,
노트북 파일을 클릭하기만 하면 그림과 숫자까지 그대로 보입니다.
본문의 어느 숫자든 그것이 나온 노트북을 열어 그 자리에서 확인하실 수 있습니다.

교재 채택을 검토하시는 강의자께 — 견본 도서를 기다리실 필요가 없습니다.
`notebooks/` 아래 아무 파일이나 눌러 보십시오.

---

## 무엇이 들어 있나

| | |
|---|---|
| 노트북 | **3판 × 35개 = 105개** (Keras 3 / TensorFlow / PyTorch) |
| 실행 결과 | **노트북 안에 담겨 있습니다** — 클릭하면 그대로 보입니다 |
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

## 장별 노트북 — 눌러서 바로 보십시오

**아무것도 설치하지 않아도 됩니다.** 아래 이름을 누르면 GitHub이 노트북을
그대로 펼쳐 보여 줍니다. 코드만이 아니라 **그때 나온 그림과 숫자까지**
담겨 있습니다. 계정도, 내려받기도 필요 없습니다.

아래는 **케라스 판** 링크입니다. 텐서플로 판과 파이토치 판은 주소에서
`keras` 를 `tensorflow` 또는 `pytorch` 로 바꾸면 됩니다. 파일 이름과 절 구성은
세 판이 같습니다.

| 장 | 노트북 | 무엇을 보이는가 |
|:--:|---|---|
| 2 | [`ch02_perceptron`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch02/ch02_perceptron.ipynb) | 델타 규칙 → 프레임워크, XOR |
| 3 | [`ch03_backend_swap`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch03/ch03_backend_swap.ipynb), [`ch03_three_frameworks`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch03/ch03_three_frameworks.ipynb) | 세 판 대조, Keras 백엔드 전환 |
| 5 | [`ch05_depth`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch05/ch05_depth.ipynb), [`ch05_init`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch05/ch05_init.ipynb), [`ch05_learning_rate`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch05/ch05_learning_rate.ipynb) | **깊이의 벽은 학습률이었다** |
| 6 | [`ch06_batchnorm`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch06/ch06_batchnorm.ipynb), [`ch06_overfitting`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch06/ch06_overfitting.ipynb), [`ch06_regularization`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch06/ch06_regularization.ipynb) | 규제, 배치 정규화의 진짜 일 |
| 7 | [`ch07_metrics`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch07/ch07_metrics.ipynb), [`ch07_selection_bias`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch07/ch07_selection_bias.ipynb) | **선택 편향은 「몇 번 골랐나」** |
| 8 | [`ch08_datasets`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch08/ch08_datasets.ipynb), [`ch08_dnn_vs_cnn`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch08/ch08_dnn_vs_cnn.ipynb), [`ch08_mnist`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch08/ch08_mnist.ipynb), [`ch08_no_activation`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch08/ch08_no_activation.ipynb) | **활성화를 빼도 최대 풀링이 비선형** |
| 9 | [`ch09_gradcam`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch09/ch09_gradcam.ipynb), [`ch09_imagenet`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch09/ch09_imagenet.ipynb), [`ch09_pretrained`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch09/ch09_pretrained.ipynb), [`ch09_transfer`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch09/ch09_transfer.ipynb) | 전이학습, Grad-CAM |
| 10 | [`ch10_forecast`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch10/ch10_forecast.ipynb), [`ch10_lr`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch10/ch10_lr.ipynb), [`ch10_memory`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch10/ch10_memory.ipynb), [`ch10_seed`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch10/ch10_seed.ipynb) | **★ 시드만 바꿔도 0.46↔1.00** |
| 11 | [`ch11_embedding_space`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch11/ch11_embedding_space.ipynb), [`ch11_representation`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch11/ch11_representation.ipynb) | **임베딩만으로는 순서를 못 본다** |
| 12 | [`ch12_attention`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch12/ch12_attention.ipynb), [`ch12_length`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch12/ch12_length.ipynb), [`ch12_lr_trap`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch12/ch12_lr_trap.ipynb), [`ch12_weights`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch12/ch12_weights.ipynb) | **어텐션도 순서를 못 본다 → 위치 부호화** |
| 13 | [`ch13_anomaly`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch13/ch13_anomaly.ipynb), [`ch13_anomaly_ood`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch13/ch13_anomaly_ood.ipynb), [`ch13_autoencoder`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch13/ch13_autoencoder.ipynb), [`ch13_denoise`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch13/ch13_denoise.ipynb) | **선형 AE = PCA (넷째 자리까지)** |
| 14 | [`ch14_gan`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch14/ch14_gan.ipynb), [`ch14_why_not_ae`](https://github.com/dhrim/deep-learning-in-one-semester/blob/main/notebooks/keras/ch14/ch14_why_not_ae.ipynb) | **손실이 오르는데 결과는 좋아진다** |

---

## 이 저장소의 규율

### ① 본문의 숫자에는 이름이 붙어 있습니다

```python
acc = metrics.accuracy(y_test, pred)
dlbook.record("ch11_lstm_acc", acc)      # 출력: ch11_lstm_acc = 0.9123
```

본문이 인용한 숫자가 **어느 노트북의 어느 줄에서 나왔는지** 이름으로 찾아갑니다.
그리고 그 줄이 노트북 출력에 그대로 남아 있습니다.

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

CPU 2코어 기준 실측으로 **35개 중 34개가 7분 안**에 끝납니다.

**`ch08_datasets` 하나만 8~11분이 걸리고, 이것은 줄이지 않습니다.**
데이터셋 셋(합성 도형·MNIST·CIFAR-10)에 DNN과 CNN을 각각 돌리는 실습입니다.
CIFAR-10이 이만큼 걸린다는 것도 학생이 겪어 봐야 할 사실입니다.
숫자로 「무겁다」고 읽는 것과 8분을 기다려 보는 것은 다릅니다.

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

# 실행하고 결과를 노트북에 담기 (출간·버전 갱신 때만)
python tools/run_notebooks.py notebooks/keras --write-back

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
