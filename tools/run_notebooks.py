#!/usr/bin/env python3
"""노트북을 실행하고, 기록된 수치를 걷어 expected.json과 대조한다.

    python tools/run_notebooks.py notebooks/keras --smoke
    python tools/run_notebooks.py notebooks/keras --check-expected

--smoke 는 DLBOOK_SMOKE=1 을 켠다 (epoch 1, 데이터 1/20).
PR마다 도는 것은 이 모드다. 전체 실행은 주 1회 야간에만 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dlbook.tracking import parse as parse_metrics  # noqa: E402

DEFAULT_TIMEOUT = 600  # 노트북 하나가 10분을 넘으면 교재용으로 부적합하다


def edition_of(nb, path: pathlib.Path) -> str:
    """이 노트북이 어느 판인가. 메타데이터 우선, 없으면 경로에서 읽는다."""
    ed = (nb.get("metadata", {}).get("dlbook", {}) or {}).get("edition")
    if ed:
        return str(ed)
    for part in path.parts:
        if part in ("keras", "tensorflow", "pytorch"):
            return part
    return "common"


def outputs_text(nb) -> str:
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", "")))
            elif out.get("output_type") == "execute_result":
                chunks.append("".join(out.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


def run_one(path: pathlib.Path, timeout: int):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    started = time.time()
    client.execute()
    return nb, time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--check-expected", action="store_true")
    ap.add_argument("--expected", default="expected.json")
    ap.add_argument("--rtol", type=float, default=0.05)
    ap.add_argument("--update-expected", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        os.environ["DLBOOK_SMOKE"] = "1"

    files: list[pathlib.Path] = []
    for p in args.paths:
        path = pathlib.Path(p)
        files.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    files = [f for f in files if ".ipynb_checkpoints" not in str(f)]

    expected_path = pathlib.Path(args.expected)
    expected = (
        json.loads(expected_path.read_text(encoding="utf-8"))
        if expected_path.exists() else {}
    )

    failures: list[str] = []
    collected: dict[str, float] = {}

    for f in files:
        label = str(f)
        try:
            nb, secs = run_one(f, args.timeout)
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 보고한다
            print(f"✗  {label}\n   {type(exc).__name__}: {str(exc)[:400]}")
            failures.append(label)
            continue

        metrics = parse_metrics(outputs_text(nb))
        ed = edition_of(nb, f)
        # 판마다 값이 다를 수 있다(초기화 방식·파라미터 셈법이 다르다).
        # 그래서 "<판>::<이름>" 으로 따로 등록한다.
        collected.update({f"{ed}::{k}": v for k, v in metrics.items()})
        note = f"  [{', '.join(metrics)}]" if metrics else ""
        print(f"✓  {label}  {secs:6.1f}초{note}")

        if secs > 600 and not args.smoke:
            print(f"   ! 10분 규칙 위반: {secs:.0f}초. 데이터나 epoch을 줄이십시오.")
            failures.append(label + " (시간 초과)")

        if args.check_expected:
            for name, value in metrics.items():
                scoped = f"{ed}::{name}"
                if scoped in expected:          # 판별 등록값이 있으면 그것부터
                    name = scoped
                if name not in expected:
                    print(f"   ! expected.json에 '{name}'이 없습니다. "
                          f"--update-expected로 등록하십시오. (측정값 {value:.4f})")
                    continue
                want = float(expected[name])
                if abs(value - want) > args.rtol * max(abs(want), 1e-9):
                    print(f"   ✗ {name}: 원고 {want:.4f} vs 실행 {value:.4f} "
                          f"(허용 오차 {args.rtol:.0%})")
                    failures.append(f"{label}::{name}")

    if args.update_expected:
        expected.update(collected)
        expected_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nexpected.json 갱신: {len(collected)}개 수치")

    print(f"\n노트북 {len(files)}개 중 실패 {len(failures)}건")
    for x in failures:
        print(f"  - {x}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
