"""dlbook — 『대학 강의를 위한 딥러닝』 공통 유틸리티.

이 패키지는 프레임워크에 의존하지 않는다.
Keras 3판·TensorFlow 판·PyTorch 판의 노트북이 **글자까지 동일하게** 쓰는
데이터 적재·시각화·평가·재현성 코드를 모아 둔 곳이다.

세 판에서 실제로 달라지는 것은 '모델 정의'와 '학습 루프' 두 셀뿐이다.
그 외의 모든 셀이 같다는 사실 자체가 이 책의 교육적 장치다.
"""

__version__ = "0.1.0"

# ── 프레임워크가 켜질 때 뱉는 잡음을 끈다 ──────────────────────────────
#
# 노트북에 실행 결과를 담아서 커밋하므로, 출력은 그대로 **교재 지면**이 된다.
# 교재를 검토하는 강의자가 노트북을 열었을 때 첫 화면이
# "This TensorFlow binary is optimized with oneAPI..." 열 줄이면 곤란하다.
#
# 이 설정은 **import tensorflow 보다 먼저** 놓여야 효과가 있다.
# 모든 노트북의 첫 코드 셀이 `import dlbook` 이므로 여기가 그 자리다.
# 잡음만 끈다. 파이썬 경고(UserWarning 등)는 건드리지 않는다 —
# 그건 학생이 봐야 할 신호일 수 있다.
import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # C++ 쪽 INFO/WARNING
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("KMP_WARNINGS", "0")           # OpenMP 잡음

from . import data, metrics, plot, repro, smoke, tracking  # noqa: F401,E402


def hush() -> None:
    """프레임워크를 불러온 뒤에도 남는 로거를 조용히 시킨다.

    환경변수로 안 잡히는 것들이 있다. absl 로거가 대표적이다.
    프레임워크를 import한 다음 한 번 부르면 된다. 안 불러도 무방하다.
    """
    import logging
    import sys

    for name in ("tensorflow", "absl", "keras", "jax._src"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if "absl" in sys.modules:
        try:
            from absl import logging as _absl
            _absl.set_verbosity(_absl.ERROR)
        except Exception:  # noqa: BLE001 — 없으면 그만이다
            pass

record = tracking.record
versions = repro.versions


def set_seed(seed: int = repro.DEFAULT_SEED, deterministic: bool = True) -> int:
    """시드를 고정하고, 그 김에 프레임워크 로거도 조용히 시킨다.

    노트북은 프레임워크를 불러온 직후 이 함수를 한 번 더 부른다(repro 참고).
    그 자리가 로거를 잡기에 가장 좋은 지점이라 여기에 붙였다.
    """
    out = repro.set_seed(seed, deterministic)
    hush()
    return out
