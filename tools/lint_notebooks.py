#!/usr/bin/env python3
"""플랫폼 중립 노트북 규칙을 강제한다.

학생 개인 노트북에 아무것도 설치하지 않고, Kaggle·JupyterHub·Colab
어디서 열어도 똑같이 도는 노트북만 저장소에 들어온다.
규칙은 `20260907_03_실습환경_검토.md`의 7원칙에서 왔다.

    python tools/lint_notebooks.py notebooks/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import sys

BANNED = [
    (r"drive\.mount\(", "R1", "구글 드라이브 마운트는 Colab 전용입니다. dlbook.data를 쓰십시오."),
    (r"(?<![\w/])/content/", "R2", "Colab 절대경로(/content/)를 쓸 수 없습니다."),
    (r"[A-Za-z]:\\\\", "R2", "윈도우 절대경로를 쓸 수 없습니다."),
    (r"/Users/|/home/[a-z]", "R2", "개인 홈 디렉터리 절대경로를 쓸 수 없습니다."),
    (r"%tensorflow_version", "R3", "%tensorflow_version은 폐기된 Colab 매직입니다."),
    (r"!\s*(wget|curl)\s", "R4", "외부에서 직접 내려받지 마십시오. dlbook.data를 통하십시오."),
    (r"!\s*apt(-get)?\s", "R5", "시스템 패키지 설치는 플랫폼마다 다릅니다."),
    (r"\.h5['\"]", "R6", "Keras 3의 모델 저장 확장자는 .keras입니다."),
]

# 설치를 허용하는 패키지 (Kaggle 기본 이미지에 없는 것 중 꼭 필요한 것만)
PIP_ALLOW = {"koreanize-matplotlib", "dlbook"}
PIP_RE = re.compile(r"!\s*pip\s+install\s+([^\n]*)")

REQUIRED_FIRST = re.compile(r"dlbook\.set_seed\s*\(|from dlbook import|import dlbook")


REQ_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def pip_targets(argline: str) -> set[str]:
    """`!pip install ...` 뒤에 오는 설치 대상의 **패키지 이름**만 뽑는다.

    `-q` 같은 옵션은 버리고, `"dlbook @ git+https://..."` 처럼 따옴표로 묶인
    PEP 508 직접 참조도 이름 `dlbook` 하나로 읽는다.
    """
    try:
        tokens = shlex.split(argline)
    except ValueError:
        tokens = argline.split()
    names: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token or token.startswith("-"):
            continue
        m = REQ_NAME.match(token)
        if m:
            names.add(m.group(0))
    return names


def cells(nb: dict) -> list[dict]:
    return nb.get("cells", [])


def source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def check(path: pathlib.Path) -> list[str]:
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: 노트북 JSON을 읽을 수 없습니다 — {exc}"]

    code = [c for c in cells(nb) if c.get("cell_type") == "code"]
    if not code:
        problems.append(f"{path}: 코드 셀이 하나도 없습니다.")

    # R7 — 첫 코드 셀에서 dlbook을 불러 시드를 고정한다
    if code and not REQUIRED_FIRST.search(source(code[0])):
        problems.append(
            f"{path}: [R7] 첫 코드 셀에서 dlbook을 불러 set_seed()를 호출해야 합니다."
        )

    # 출력은 커밋하지 않는다 — diff를 읽을 수 있게 유지한다
    for i, c in enumerate(code):
        if c.get("outputs") or c.get("execution_count") is not None:
            problems.append(
                f"{path}: [R8] 코드 셀 {i + 1}에 실행 결과가 남아 있습니다. "
                "커밋 전에 출력을 지우십시오."
            )
            break

    for i, c in enumerate(cells(nb)):
        src = source(c)
        for pattern, rule, msg in BANNED:
            if re.search(pattern, src):
                problems.append(f"{path}: [{rule}] 셀 {i + 1} — {msg}")
        for m in PIP_RE.finditer(src):
            bad = pip_targets(m.group(1)) - PIP_ALLOW
            if bad:
                problems.append(
                    f"{path}: [R5] 셀 {i + 1} — 허용되지 않은 설치: {', '.join(sorted(bad))}"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    files: list[pathlib.Path] = []
    for p in args.paths:
        path = pathlib.Path(p)
        files.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    files = [f for f in files if ".ipynb_checkpoints" not in str(f)]

    all_problems = [p for f in files for p in check(f)]
    for problem in all_problems:
        print(problem)
    print(f"\n노트북 {len(files)}개 검사, 문제 {len(all_problems)}건")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
