"""원고가 인용하는 수치를 기계가 검증할 수 있게 남긴다.

노트북에서

    dlbook.record("mnist_cnn_test_acc", acc)

라고 쓰면, 화면에는 사람이 읽을 한 줄이 찍히고
출력에는 기계가 읽을 표식이 함께 남는다.

CI가 그 표식을 걷어 `expected.json`과 대조한다.
**원고에 적힌 "정확도 0.982"가 지금도 사실인지를 매주 기계가 확인한다.**
손으로 옮겨 적은 수치는 반드시 언젠가 거짓이 된다.
"""

from __future__ import annotations

import json

MARKER = "##DLBOOK_METRIC##"


def record(name: str, value, digits: int = 4) -> float:
    """수치 하나를 기록하고 그대로 돌려준다.

    Parameters
    ----------
    name:
        원고에서 이 수치를 부르는 이름. 장·주제를 담아 짓는다.
        예: "ch08_cnn_mnist_test_acc"
    value:
        float로 변환 가능한 값.
    """
    v = float(value)
    print(f"{name} = {v:.{digits}f}")
    print(MARKER + json.dumps({"name": name, "value": v}, ensure_ascii=False))
    return v


def parse(text: str) -> dict[str, float]:
    """실행된 노트북 출력 문자열에서 기록된 수치들을 걷어 온다."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(MARKER):
            continue
        try:
            item = json.loads(line[len(MARKER):])
            out[item["name"]] = float(item["value"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out
