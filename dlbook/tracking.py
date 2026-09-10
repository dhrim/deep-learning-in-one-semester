"""원고가 인용하는 수치에 이름을 붙여 출력한다.

노트북에서

    dlbook.record("mnist_cnn_test_acc", acc)

라고 쓰면 출력에 한 줄이 찍힌다.

    mnist_cnn_test_acc = 0.9512

본문이 "정확도 0.95" 라고 적을 때, 그 숫자가 **어느 노트북의 어느 줄에서
나온 것인지** 이름으로 바로 찾을 수 있게 하는 것이 전부다.

예전에는 이 함수가 기계가 읽을 표식을 함께 뱉었고, CI가 그것을 걷어
`expected.json` 과 대조했다. 그 장치는 폐기했다.
노트북에 실행 결과를 담아 커밋하기로 했으므로, 본문 수치가 맞는지는
**노트북을 열어 보면 된다.** 기계가 대조할 이유가 없다.
게다가 딥러닝은 돌릴 때마다 값이 달라서, 자동 대조는 원리적으로
가짜 경보를 낼 수밖에 없었다.
"""

from __future__ import annotations


def record(name: str, value, digits: int = 4) -> float:
    """수치 하나에 이름을 붙여 찍고, 값을 그대로 돌려준다.

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
    return v
