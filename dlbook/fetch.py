"""실습 데이터셋을 미리 내려받는다.

    python -m dlbook.fetch                    # ~/.dlbook/datasets 에 받는다
    python -m dlbook.fetch --dest data/       # 저장소의 data/ 에 받는다
    python -m dlbook.fetch --weights          # ImageNet 사전학습 가중치까지

**강의 전날 한 번 돌려 두는 것을 권합니다.** 강의실에서 내려받기가 막혀도
실습이 그대로 돕니다. 받은 폴더를 통째로 옮겨서 쓰려면 `DLBOOK_DATA` 가
그 폴더를 가리키게 하십시오.

받는 것:

| 파일 | 크기 |
|---|---|
| `mnist.npz` | 11 MB |
| `fashion-mnist/*.gz` (4개) | 30 MB |
| `cifar-10-python.tar.gz` | 170 MB |
| `--weights` 를 주면 VGG16 · ResNet50 · MobileNetV2 · resnet18 | 210 MB |
"""

from __future__ import annotations

import argparse
import sys

from . import data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dlbook.fetch",
        description="실습 데이터셋을 미리 내려받습니다.",
    )
    ap.add_argument("--dest", default=None,
                    help="둘 폴더 (기본: ~/.dlbook/datasets)")
    ap.add_argument("--weights", action="store_true",
                    help="ImageNet 사전학습 가중치까지 받습니다 (약 210MB)")
    ap.add_argument("--quiet", action="store_true", help="조용히")
    a = ap.parse_args(argv)

    try:
        data.fetch_all(dest=a.dest, weights=a.weights, quiet=a.quiet)
    except Exception as exc:
        print(f"\n실패했습니다: {exc}", file=sys.stderr)
        print("인터넷이 막힌 곳이라면, 다른 곳에서 받아 옮기십시오.", file=sys.stderr)
        print("어디에 두면 되는지는 data/README.md 에 있습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
