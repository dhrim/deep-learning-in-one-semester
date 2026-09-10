#!/usr/bin/env python3
"""노트북이 어디서 열어도 돌아가는지 검사한다.

학생 개인 노트북에 아무것도 설치하지 않고, Kaggle·JupyterHub·Colab
어디서 열어도 똑같이 도는 노트북만 저장소에 들어온다.

    R1  google.colab 을 import 하지 않는다
    R2  드라이브 마운트를 하지 않는다
    R3  절대 경로(/content, /kaggle)를 쓰지 않는다
    R4  pip install 은 부트스트랩 셀에서만
    R5  데이터는 dlbook.data 로만 얻는다
    R6  시드를 dlbook.set_seed() 로 고정한다
    R7  수치는 dlbook.record() 로 남긴다          (경고)
    R8  한글 폰트는 plot.use_korean() 으로
    -   모델은 .keras 로 저장한다 (Keras 3)
    -   실행 결과를 담아서 커밋한다               (경고)

**이 R 번호는 이 파일 안에서만 쓰는 이름이다.** 책에는 같은 규칙들이
번호 없이 실려 있다(3장, 부록 B.2). 규칙 번호는 원고와 코드를 만드는
쪽의 사정이지 독자가 알아야 할 것이 아니라서, 책에서 뺐다.
그러니 여기 번호를 바꾸어도 책은 고칠 것이 없다. 규칙의 **내용**이
달라질 때만 양쪽을 함께 본다.

오류가 하나라도 있으면 실패한다. 경고는 알리기만 하고 실패시키지 않는다.

    python tools/lint_notebooks.py notebooks/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import sys

# ── 소스에 나오면 안 되는 것 ────────────────────────────────────────
BANNED = [
    (r"(?:import|from)\s+google\.colab", "R1",
     "google.colab 은 Colab 밖에서 죽습니다."),
    (r"%tensorflow_version", "R1",
     "%tensorflow_version 은 Colab 전용 매직이고, 그나마 폐기됐습니다."),

    (r"drive\.mount\(", "R2",
     "구글 드라이브 마운트는 Colab 전용입니다. dlbook.data 를 쓰십시오."),

    (r"(?<![\w/])/content/", "R3", "Colab 절대경로(/content/)를 쓸 수 없습니다."),
    (r"(?<![\w/])/kaggle/", "R3", "Kaggle 절대경로(/kaggle/)를 쓸 수 없습니다."),
    (r"[A-Za-z]:\\\\", "R3", "윈도우 절대경로를 쓸 수 없습니다."),
    (r"/Users/|/home/[a-z]", "R3", "개인 홈 디렉터리 절대경로를 쓸 수 없습니다."),

    (r"!\s*apt(-get)?\s", "R4", "시스템 패키지 설치는 플랫폼마다 다릅니다."),

    (r"!\s*(wget|curl)\s", "R5",
     "외부에서 직접 내려받지 마십시오. dlbook.data 를 통하십시오."),

    (r"\.h5['\"]", "저장형식", "Keras 3의 모델 저장 확장자는 .keras 입니다."),
]

# 설치를 허용하는 패키지 (Kaggle 기본 이미지에 없는 것 중 꼭 필요한 것만)
PIP_ALLOW = {"koreanize-matplotlib", "dlbook"}
PIP_RE = re.compile(r"!\s*pip\s+install\s+([^\n]*)")

SEED_RE = re.compile(r"dlbook\.set_seed\s*\(|from dlbook import|import dlbook")
RECORD_RE = re.compile(r"\brecord\s*\(")
KOREAN_RE = re.compile(r"use_korean\s*\(")
PLOT_RE = re.compile(r"\bplt\.|matplotlib|\bplot\.")

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


def check(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """(오류, 경고)를 돌려준다. 오류가 하나라도 있으면 검사는 실패한다."""
    problems: list[str] = []
    warnings: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: 노트북 JSON을 읽을 수 없습니다 — {exc}"], []

    code = [c for c in cells(nb) if c.get("cell_type") == "code"]
    if not code:
        problems.append(f"{path}: 코드 셀이 하나도 없습니다.")
        return problems, warnings

    sources = [source(c) for c in code]
    whole = "\n".join(sources)

    # ── 금지 문자열 ────────────────────────────────────────────────
    for i, c in enumerate(cells(nb)):
        src = source(c)
        for pattern, rule, msg in BANNED:
            if re.search(pattern, src):
                problems.append(f"{path}: [{rule}] 셀 {i + 1} — {msg}")

    # ── R4 — pip install 은 부트스트랩 셀(첫 코드 셀)에서만 ────────
    #
    # 중간에서 설치하면 이미 불러온 모듈과 충돌한다. numpy 계열이 특히 그렇다.
    # 학생 절반이 여기서 막힌다.
    for i, src in enumerate(sources):
        for m in PIP_RE.finditer(src):
            if i != 0:
                problems.append(
                    f"{path}: [R4] 코드 셀 {i + 1} — pip install 은 첫 코드 셀"
                    "(부트스트랩)에서만 합니다."
                )
            bad = pip_targets(m.group(1)) - PIP_ALLOW
            if bad:
                problems.append(
                    f"{path}: [R4] 코드 셀 {i + 1} — 허용되지 않은 설치: "
                    f"{', '.join(sorted(bad))}"
                )

    # ── R6 — 첫 코드 셀에서 dlbook 을 불러 시드를 고정한다 ─────────
    if not SEED_RE.search(sources[0]):
        problems.append(
            f"{path}: [R6] 첫 코드 셀에서 dlbook 을 불러 set_seed() 를 "
            "호출해야 합니다."
        )

    # ── R8 — 그림을 그리면 한글 폰트를 잡는다 ──────────────────────
    if PLOT_RE.search(whole) and not KOREAN_RE.search(whole):
        problems.append(
            f"{path}: [R8] 그림을 그리는데 plot.use_korean() 이 없습니다. "
            "플랫폼마다 폰트가 달라 한글이 깨집니다."
        )

    # ── R7 — 수치는 record() 로 남긴다 (경고) ──────────────────────
    #
    # 오류로 두지 않는다. 가중치를 그림으로만 보이는 노트북처럼
    # 인용할 수치가 없는 것도 있다.
    if not RECORD_RE.search(whole):
        warnings.append(
            f"{path}: [R7] dlbook.record() 가 하나도 없습니다. "
            "본문이 인용할 수치가 있다면 이름을 붙여 남기십시오."
        )

    # ── 실행 결과를 담아서 커밋한다 (경고) ─────────────────────────
    #
    # 예전에는 정반대였다. 출력이 남아 있으면 실패시켰다.
    # 그러면 저장소를 열어 본 사람은 코드만 보고 결과는 못 본다.
    # 강의 교재에서 그건 치명적이다 — 교재를 고르는 강의자가 보고 싶은 것은
    # `model.fit(...)` 이라는 코드가 아니라 그것을 돌리면 나오는 그림과 숫자다.
    #
    # 출력을 담아 두면 GitHub이 노트북을 그대로 렌더링하므로,
    # 아무것도 설치하지 않고 링크 하나로 실습 전체를 볼 수 있다.
    if not any(c.get("outputs") for c in code):
        warnings.append(
            f"{path}: 실행 결과가 하나도 없습니다. "
            "돌린 뒤 출력을 담아 커밋하십시오 "
            "(python tools/run_notebooks.py <경로> --write-back)."
        )

    return problems, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    files: list[pathlib.Path] = []
    for p in args.paths:
        path = pathlib.Path(p)
        files.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    files = [f for f in files if ".ipynb_checkpoints" not in str(f)]

    all_problems: list[str] = []
    all_warnings: list[str] = []
    for f in files:
        problems, warnings = check(f)
        all_problems.extend(problems)
        all_warnings.extend(warnings)

    for problem in all_problems:
        print(problem)
    for warning in all_warnings:
        print("경고 — " + warning)

    print(
        f"\n노트북 {len(files)}개 검사, "
        f"문제 {len(all_problems)}건, 경고 {len(all_warnings)}건"
    )
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
