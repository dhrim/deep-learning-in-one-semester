"""dlbook — 『대학 강의를 위한 딥러닝』 공통 유틸리티.

이 패키지는 프레임워크에 의존하지 않는다.
Keras 3판·TensorFlow 판·PyTorch 판의 노트북이 **글자까지 동일하게** 쓰는
데이터 적재·시각화·평가·재현성 코드를 모아 둔 곳이다.

세 판에서 실제로 달라지는 것은 '모델 정의'와 '학습 루프' 두 셀뿐이다.
그 외의 모든 셀이 같다는 사실 자체가 이 책의 교육적 장치다.
"""

__version__ = "0.1.0"

from . import data, metrics, plot, repro, smoke, tracking  # noqa: F401,E402

set_seed = repro.set_seed
record = tracking.record
versions = repro.versions
