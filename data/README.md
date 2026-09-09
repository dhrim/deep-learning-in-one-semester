# data/ — 데이터셋을 두는 곳

**이 폴더의 데이터 파일은 git에 올라가지 않습니다.** (`.gitignore`)
전부 **코드가 내려받습니다.** 아무것도 미리 넣어 둘 필요가 없습니다.

```python
from dlbook import data
s = data.mnist()        # 없으면 받아서 캐시에 둔다
```

캐시는 `~/.dlbook/datasets` 입니다.

---

## 강의 전날 한 번 — 미리 받아 두기

강의실 네트워크가 막혀 있거나 느릴 때를 대비해, **한 번에 다 받아 두는 명령**이 있습니다.

```bash
python -m dlbook.fetch                  # ~/.dlbook/datasets 에
python -m dlbook.fetch --dest data/     # 이 폴더에
python -m dlbook.fetch --weights        # ImageNet 사전학습 가중치까지 (+210MB)
```

이미 있는 파일은 건너뜁니다. 받은 폴더를 USB로 옮겨 쓰려면:

```bash
export DLBOOK_DATA=/옮긴/폴더
```

---

## 받는 것

| 파일 | 크기 | 쓰는 곳 | 받는 곳 |
|---|:--:|---|---|
| `mnist.npz` | 11 MB | 8장 이후 | https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz |
| `fashion-mnist/*.gz` (4개) | 30 MB | 8·13·14장 | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ |
| `cifar-10-python.tar.gz` | 170 MB | 9장 | https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz |
| `pretrained/` (ImageNet 가중치) | 210 MB | 9장 전이학습 | 케라스·파이토치가 스스로 받습니다 |

**1~7장은 아무것도 내려받지 않습니다.** 귤·사과·XOR·나선·도형·기억 과제·영화평은
전부 `dlbook.data` 가 **그 자리에서 만들어 내는** 합성 데이터입니다.
첫날 실습이 네트워크에 걸려 멈추는 일은 없습니다.

---

## 찾는 순서

`dlbook.data` 는 **로컬을 먼저 뒤진 뒤** 없으면 내려받습니다.

1. 환경변수 `DLBOOK_DATA` 가 가리키는 폴더
2. **이 폴더** (`<저장소>/data/`)
3. 현재 작업 폴더의 `data/`
4. `~/.dlbook/datasets` (내려받은 것을 두는 캐시)

하위 폴더 이름(`fashion-mnist/…`)으로 찾다가 없으면 **파일 이름만으로 한 번 더** 찾습니다.
폐쇄망에서 파일을 폴더 없이 통째로 부어 넣어도 됩니다.

```python
from dlbook import data
data.data_dirs()                              # 어디를 뒤지는지
data.find_local("mnist.npz")                  # 찾았는지
data.pretrained_dir()                         # 가중치를 두는 곳
```

---

## 사전학습 가중치 (9장)

케라스는 `$KERAS_HOME/models/`, 파이토치는 `$TORCH_HOME/hub/checkpoints/` 를 봅니다.
`dlbook` 을 import 할 때 **`data/pretrained/` 폴더가 이미 있으면 두 환경변수를 그쪽으로
맞춥니다.** 그래서 이 폴더만 복사해 가면 인터넷 없이 전이학습 실습이 돕니다.

```
data/pretrained/
├── models/                  ← 케라스 (.h5)
└── hub/checkpoints/         ← 파이토치 (.pth)
```

폴더가 없으면 아무것도 하지 않습니다. 각 라이브러리의 기본 경로(`~/.keras`, `~/.cache/torch`)를 씁니다.

---

## 왜 이렇게까지 하는가

이 책을 쓰는 환경에서 데이터셋 내려받기가 프록시에 막혀 있었습니다.
드문 일이 아닙니다 — 기업 교육장, 공공기관, 대학 전산실에서 흔히 겪습니다.

**그런 곳에서도 이 책의 실습이 전부 돌아가야 합니다.**
합성 데이터를 따로 만든 것도, 미리 받아 두는 명령을 둔 것도 같은 이유입니다.
