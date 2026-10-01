#!/usr/bin/env python3
"""Prove verify_lifecycle_report.py actually catches corruption.

A verifier that has only ever seen a clean document is untested. Each tamper
below edits exactly one published fact; the verifier must fail on every one
and must return to a clean pass once the document is restored.

File-based, not an inline heredoc: a heredoc steals stdin and the harness
silently no-ops.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

FIN = pathlib.Path(__file__).resolve().parent.parent
DOC = FIN / "report" / "LIFECYCLE-COST.md"
VERIFY = FIN / "tools" / "verify_lifecycle_report.py"
PY = "/tmp/dlcvenv/bin/python"

# (label, find, replace) -- each must break exactly one published claim.
TAMPERS = [
    ("init time inflated",
     "| 2 | NCCL | 1905.3 |", "| 2 | NCCL | 9905.3 |"),
    ("steady op altered",
     "| 2 | torch.distributed | 1630.2 | 0.3 | 230.2 |",
     "| 2 | torch.distributed | 1630.2 | 0.3 | 130.2 |"),
    ("transport relabelled",
     "| 2 | NCCL | 1382.0 | 281.2 | 902.0 | 152.6 | Libfabric |",
     "| 2 | NCCL | 1382.0 | 281.2 | 902.0 | 152.6 | Socket |"),
    ("destroy cost hidden",
     "| 4 | NCCL | 2483.9 | 656.3 | 107.5 | 485.8 |",
     "| 4 | NCCL | 2483.9 | 656.3 | 107.5 | 4.8 |"),
    ("failure disclosure removed",
     "cxil_map: write error", "some other error"),
    ("threshold failing-range corrupted",
     "256 KiB through 16 MiB all fail",
     "999 KiB through 16 MiB all fail"),
    ("threshold passing-sizes corrupted",
     "4 KiB and 64 KiB pass",
     "4 MiB and 64 MiB pass"),
    ("root-cause claim smuggled in",
     "Not root-caused", "Fully root-caused"),
]


def run_verifier():
    r = subprocess.run([PY, str(VERIFY)], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    if not DOC.exists():
        print(f"missing {DOC}", file=sys.stderr)
        return 2

    rc, out = run_verifier()
    if rc != 0:
        print("clean document does not pass; fix that first")
        print(out)
        return 2
    print(f"baseline: clean document passes ({out.splitlines()[0]})")

    backup = pathlib.Path(tempfile.gettempdir()) / "LIFECYCLE-COST.backup.md"
    shutil.copy(DOC, backup)
    original = DOC.read_text()

    caught = 0
    try:
        for label, find, repl in TAMPERS:
            if find not in original:
                print(f"  SKIP  {label}: anchor not present")
                continue
            DOC.write_text(original.replace(find, repl, 1))
            rc, out = run_verifier()
            first = out.splitlines()[0] if out else ""
            if rc != 0:
                print(f"  CAUGHT {label}  ({first})")
                caught += 1
            else:
                print(f"  MISSED {label}  <-- verifier is blind to this")
            DOC.write_text(original)
    finally:
        DOC.write_text(original)
        shutil.copy(backup, backup.with_suffix(".kept"))

    rc, out = run_verifier()
    print(f"restored: rc={rc} ({out.splitlines()[0] if out else ''})")

    total = sum(1 for _, f, _ in TAMPERS if f in original)
    print(f"\n{caught}/{total} tampers caught")
    return 0 if (caught == total and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
