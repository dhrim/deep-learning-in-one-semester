"""재현성 — 시드 하나로 파이썬·numpy·TensorFlow·PyTorch를 동시에 고정한다.

기존 240개 노트북 중 81개에 시드 고정이 없었다.
같은 코드를 두 번 돌려 다른 숫자가 나오면 학생은 자기가 무엇을 잘못했는지
판단할 수 없다. 이 책의 모든 노트북은 첫 셀에서 set_seed()를 부른다.
"""

from __future__ import annotations

import os
import random
import sys

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """이미 불러온 프레임워크의 난수 시드를 모두 고정한다.

    **아직 import되지 않은 프레임워크는 건드리지 않는다.** PyTorch 판 노트북에서
    이 함수가 TensorFlow를 끌어오면 쓰지도 않을 라이브러리 때문에 메모리가
    수 GB 늘고, 브라우저 실습 환경에서는 그것만으로 커널이 죽는다.

    그래서 **프레임워크를 import한 다음 한 번 더 부른다.** 각 판의 노트북에서
    모델을 만들기 직전에 `dlbook.set_seed(42)` 가 다시 나오는 이유다.

    Parameters
    ----------
    seed:
        고정할 시드 값.
    deterministic:
        True이면 GPU 연산까지 결정적으로 만든다(느려질 수 있다).

    Returns
    -------
    적용된 시드 값.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)

    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if "keras" in sys.modules:
        sys.modules["keras"].utils.set_random_seed(seed)

    return seed


def backend() -> str:
    """현재 Keras 백엔드 이름.

    Keras를 아직 불러오지 않았으면 앞으로 쓰일 값을 돌려준다.
    KERAS_BACKEND 가 있으면 그것, 없으면 Keras 3의 기본값인 tensorflow.
    """
    if "keras" not in sys.modules:
        # 파이토치 판에는 Keras가 아예 없다. 그런 환경에서 "tensorflow"라고
        # 답하면 노트북 첫 줄이 거짓말을 한다.
        if _installed_version("keras") == "-":
            return "-"
        return os.environ.get("KERAS_BACKEND", "tensorflow")
    return sys.modules["keras"].backend.backend()


# 배포 이름이 import 이름과 다른 것들. tensorflow는 tensorflow-cpu로 깔린다.
_DIST_NAMES = {
    "tensorflow": ("tensorflow", "tensorflow-cpu", "tensorflow-macos"),
    "torch": ("torch",),
    "keras": ("keras",),
    "numpy": ("numpy",),
}


def _installed_version(name: str) -> str:
    """**불러오지 않고** 설치된 버전만 읽는다.

    importlib.metadata 는 패키지 메타데이터만 보므로 모듈을 메모리에
    올리지 않는다. 쓰지도 않을 라이브러리로 수 GB를 쓰는 일이 없다.
    """
    from importlib.metadata import PackageNotFoundError, version

    for dist in _DIST_NAMES.get(name, (name,)):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return "-"


def versions() -> dict[str, str]:
    """이 노트북이 돌아간 환경의 버전을 모아 반환한다.

    이미 불러온 것은 모듈에서 직접 읽고, 아직 안 불러온 것은
    **설치된 버전**을 메타데이터에서 읽는다(모듈을 올리지 않는다).
    설치돼 있지 않으면 "-".

    노트북에 실행 결과를 담아 커밋하므로 이 한 줄은 그대로 교재에 실린다.
    "이 결과는 어느 버전에서 나온 것인가"에 답하는 줄이라 비어 있으면 안 된다.
    """
    import platform

    out = {"python": platform.python_version()}
    for name in ("numpy", "keras", "tensorflow", "torch"):
        mod = sys.modules.get(name)
        out[name] = (
            getattr(mod, "__version__", "?") if mod else _installed_version(name)
        )
    out["keras_backend"] = backend()
    return out
