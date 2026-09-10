#!/usr/bin/env python3
"""노트북을 실행한다. 그리고 원하면 실행 결과를 그 노트북에 담아 둔다.

    python tools/run_notebooks.py notebooks/keras --smoke
    python tools/run_notebooks.py notebooks/keras --write-back

--smoke 는 DLBOOK_SMOKE=1 을 켠다 (epoch 1, 데이터 1/20).
노트북을 고쳤을 때 "돌아가기는 하는가"만 빠르게 보는 모드다.

--write-back 은 실행된 결과를 노트북 파일에 그대로 저장한다.
저장소를 열어 본 사람이 아무것도 설치하지 않고 결과를 볼 수 있게 하는 것이
이 저장소의 전제다. GitHub이 .ipynb를 그대로 렌더링하므로,
출력이 담겨 있으면 링크 하나로 실습 전체가 보인다.

--smoke 와 --write-back 은 함께 쓸 수 없다.
축소 실행의 결과를 교재에 담아 두면 거짓말이 되기 때문이다.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_TIMEOUT = 600  # 노트북 하나가 10분을 넘으면 교재용으로 부적합하다

# 저장할 때 걷어내는 잡음.
#
# 프레임워크가 켜지면서 C++ 쪽에서 stderr로 직접 쏟는 줄들이다.
# 환경변수(TF_CPP_MIN_LOG_LEVEL 등)로는 안 잡힌다. absl 로거가 켜지기
# **전에** 찍히기 때문이다. 파이썬 계층에서 막을 방법이 없다.
#
# 노트북에 실행 결과를 담아 커밋하므로 이 출력은 그대로 교재 지면이 된다.
# 교재를 검토하러 노트북을 연 강의자의 첫 화면이 oneDNN 안내문이면 곤란하다.
# 계산 결과가 아니라 **실행기가 남긴 부스러기**만 지운다.
# 경고든 오류든 코드가 낸 것은 하나도 건드리지 않는다.
NOISE = (
    "absl::InitializeLog",
    "oneDNN custom operations are on",
    "[IPKernelApp]",
    "Kernel is running over TCP",
    "MissingIDFieldWarning",
    "external/local_xla",
    "TF-TRT Warning",
)


def strip_noise(nb) -> int:
    """실행기가 남긴 잡음 줄을 걷어내고, 지운 줄 수를 돌려준다."""
    removed = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        kept_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream" or out.get("name") != "stderr":
                kept_outputs.append(out)
                continue
            text = out.get("text", "")
            lines = text if isinstance(text, list) else text.splitlines(keepends=True)
            kept = [ln for ln in lines if not any(n in ln for n in NOISE)]
            removed += len(lines) - len(kept)
            if not "".join(kept).strip():
                continue                      # 통째로 잡음이면 출력 자체를 뺀다
            out["text"] = kept
            kept_outputs.append(out)
        cell["outputs"] = kept_outputs
    return removed


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
    ap.add_argument(
        "--write-back", action="store_true",
        help="실행 결과를 노트북 파일에 담아 저장한다",
    )
    ap.add_argument(
        "--skip-executed", action="store_true",
        help="이미 결과가 담긴 노트북은 건너뛴다 (중단된 갱신을 이어서 할 때)",
    )
    args = ap.parse_args()

    if args.smoke and args.write_back:
        print("--smoke 와 --write-back 은 함께 쓸 수 없습니다. "
              "축소 실행 결과를 교재에 담으면 거짓이 됩니다.")
        return 2

    if args.smoke:
        os.environ["DLBOOK_SMOKE"] = "1"

    files: list[pathlib.Path] = []
    for p in args.paths:
        path = pathlib.Path(p)
        files.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    files = [f for f in files if ".ipynb_checkpoints" not in str(f)]

    if args.skip_executed:
        import json as _json

        def done(p: pathlib.Path) -> bool:
            nb = _json.loads(p.read_text(encoding="utf-8"))
            return any(
                c.get("outputs")
                for c in nb.get("cells", [])
                if c.get("cell_type") == "code"
            )

        before = len(files)
        files = [f for f in files if not done(f)]
        skipped = before - len(files)
        if skipped:
            print(f"이미 결과가 담긴 {skipped}개는 건너뜁니다. 남은 것 {len(files)}개.")

    failures: list[str] = []
    written = 0

    for f in files:
        label = str(f)
        try:
            nb, secs = run_one(f, args.timeout)
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 보고한다
            print(f"✗  {label}\n   {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            failures.append(label)
            continue

        note = ""
        if args.write_back:
            import nbformat
            cut = strip_noise(nb)
            nbformat.write(nb, f)
            written += 1
            note = f"  → 저장 {f.stat().st_size / 1024:.0f}KB"
            if cut:
                note += f" (잡음 {cut}줄 제거)"

        print(f"✓  {label}  {secs:6.1f}초{note}", flush=True)

        # 10분 규칙 — 실습 60분 안에 「돌리고 → 바꿔서 다시 돌리고 → 견주기」가
        # 들어가야 한다. 다만 이것으로 실행을 실패시키지는 않는다.
        # ch08_datasets 는 8~11분이 걸리고, 그건 줄이지 않기로 했다.
        # CIFAR-10이 이만큼 걸린다는 것도 학생이 겪어 봐야 할 사실이기 때문이다.
        if secs > 600 and not args.smoke:
            print(f"   ! 10분을 넘었습니다: {secs:.0f}초. 의도한 것이 아니라면 "
                  "데이터나 epoch을 줄이십시오.")

    if args.write_back:
        print(f"\n결과를 담아 저장한 노트북 {written}개")

    print(f"\n노트북 {len(files)}개 중 실패 {len(failures)}건")
    for x in failures:
        print(f"  - {x}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
