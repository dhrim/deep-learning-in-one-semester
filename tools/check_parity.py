#!/usr/bin/env python3
"""세 판의 대응이 유지되고 있는지 확인한다.

Keras 3판이 정본이다. TensorFlow 판과 PyTorch 판은
**같은 파일 이름**으로 존재해야 하고, **셀 개수와 마크다운 제목이 같아야** 한다.
세 판을 나란히 놓고 읽는 것이 이 책의 교육적 장치이므로,
한쪽만 조용히 달라지면 그 장치가 무너진다.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
EDITIONS = ("keras", "tensorflow", "pytorch")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$", re.M)


def headings(path: pathlib.Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        out += [f"{m.group(1)} {m.group(2).strip()}" for m in HEADING.finditer(src)]
    return out


def main() -> int:
    master = NB / EDITIONS[0]
    if not master.exists():
        print("notebooks/keras 가 없습니다.")
        return 1

    problems: list[str] = []
    rels = sorted(
        p.relative_to(master)
        for p in master.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in str(p)
    )

    for rel in rels:
        want = headings(master / rel)
        for edition in EDITIONS[1:]:
            other = NB / edition / rel
            if not other.exists():
                problems.append(f"{edition}/{rel} 가 없습니다. Keras 판에는 있습니다.")
                continue
            got = headings(other)
            if got != want:
                only_a = [h for h in want if h not in got]
                only_b = [h for h in got if h not in want]
                problems.append(
                    f"{edition}/{rel} 의 절 구성이 Keras 판과 다릅니다.\n"
                    + (f"    Keras에만: {only_a}\n" if only_a else "")
                    + (f"    {edition}에만: {only_b}" if only_b else "")
                )

    for edition in EDITIONS[1:]:
        base = NB / edition
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in str(p):
                continue
            rel = p.relative_to(base)
            if not (master / rel).exists():
                problems.append(f"{edition}/{rel} 에 대응하는 Keras 판이 없습니다.")

    for x in problems:
        print(x)
    print(f"\n정본 노트북 {len(rels)}개, 대응 문제 {len(problems)}건")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
