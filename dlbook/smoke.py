"""스모크 모드 — CI가 노트북을 빠르게 완주시키기 위한 축소 실행.

환경변수 DLBOOK_SMOKE=1 이면
  * epoch 수를 1로 줄이고
  * 데이터를 1/20로 잘라 쓴다.

학생이 노트북을 열었을 때는 이 변수가 없으므로 원래 설정으로 돈다.
즉 '학생이 보는 코드'와 'CI가 도는 코드'가 같은 파일이다.
"""

from __future__ import annotations

import os

SUBSET_DIVISOR = 20


def is_smoke() -> bool:
    return os.environ.get("DLBOOK_SMOKE", "") not in ("", "0", "false", "False")


def epochs(n: int) -> int:
    """스모크 모드이면 1, 아니면 n."""
    return 1 if is_smoke() else n


def subset(*arrays, divisor: int = SUBSET_DIVISOR):
    """스모크 모드이면 배열들을 앞에서부터 1/divisor 만큼만 잘라 돌려준다.

    잘라내는 길이는 모든 배열에 대해 동일하므로 입력·레이블 짝이 유지된다.
    """
    if not is_smoke():
        return arrays if len(arrays) > 1 else arrays[0]

    n = min(len(a) for a in arrays)
    k = max(1, n // divisor)
    cut = tuple(a[:k] for a in arrays)
    return cut if len(cut) > 1 else cut[0]
