"""dlbook 단위 테스트 — 프레임워크 없이 도는 것만 담는다."""

import numpy as np
import pytest

import dlbook
from dlbook import data, metrics


def test_split_is_disjoint():
    x, y = data.apples(500)
    s = data.split(x, y, val_ratio=0.2, test_ratio=0.2, seed=1)
    assert len(s.x_train) + len(s.x_val) + len(s.x_test) == 500
    rows = {tuple(r) for r in s.x_train}
    assert not rows & {tuple(r) for r in s.x_val}
    assert not rows & {tuple(r) for r in s.x_test}


def test_split_rejects_impossible_ratios():
    x, y = data.apples(50)
    with pytest.raises(ValueError):
        data.split(x, y, val_ratio=0.6, test_ratio=0.6)


def test_split_is_reproducible():
    x, y = data.apples(300)
    a = data.split(x, y, seed=7)
    b = data.split(x, y, seed=7)
    assert np.array_equal(a.x_train, b.x_train)


def test_apples_boundary_is_learnable():
    """사과 데이터는 x1 + x2 = 3.5 로 대부분 갈려야 한다 (2.5절의 그 선)."""
    x, y = data.apples(2000, noise=0.35)
    pred = (x[:, 0] + x[:, 1] - 3.5 > 0).astype(int)
    assert metrics.accuracy(y, pred) > 0.85


def test_xor_is_not_linearly_separable():
    """XOR은 어떤 직선으로도 90%를 넘길 수 없다 — 2.13절의 논거."""
    x, y = data.xor(2000, noise=0.15)
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(3000):
        w = rng.normal(size=2)
        b = rng.normal()
        acc = metrics.accuracy(y, (x @ w + b > 0).astype(int))
        best = max(best, acc, 1 - acc)
    assert best < 0.9, f"직선 하나로 {best:.3f}까지 갈렸습니다. XOR 데이터를 확인하십시오."


def test_confusion_matrix_orientation():
    """행=실제, 열=예측. 이 방향이 바뀌면 정밀도와 재현율이 뒤집힌다."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = metrics.confusion_matrix(y_true, y_pred)
    assert cm[0, 1] == 1  # 실제 0인데 1로 예측 = 거짓 양성
    assert cm[1, 0] == 0


def test_auc_of_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert metrics.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_smoke_mode_shrinks(monkeypatch):
    x, y = data.apples(1000)
    monkeypatch.setenv("DLBOOK_SMOKE", "1")
    xs, ys = dlbook.smoke.subset(x, y)
    assert len(xs) == len(ys) == 50
    assert dlbook.smoke.epochs(30) == 1
    monkeypatch.delenv("DLBOOK_SMOKE")
    assert dlbook.smoke.epochs(30) == 30


def test_record_prints_named_value(capsys):
    """record 는 이름을 붙여 한 줄 찍고 값을 그대로 돌려준다.

    이 한 줄이 노트북 출력에 남아 커밋되므로, 본문이 인용한 숫자가
    어느 노트북의 어느 줄에서 나왔는지 이름으로 찾을 수 있다.
    """
    assert dlbook.record("ch02_apple_test_acc", 0.9123) == 0.9123
    assert capsys.readouterr().out.strip() == "ch02_apple_test_acc = 0.9123"


def test_set_seed_returns_seed():
    assert dlbook.set_seed(123) == 123
