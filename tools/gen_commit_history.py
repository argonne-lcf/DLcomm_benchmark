#!/usr/bin/env python3
"""Regenerate docs/COMMITS.md from git itself.

The commit history is generated rather than written by hand for the same
reason `docs/INDEX.md` is: a hand-maintained changelog drifts from the
repository as soon as a commit is amended, reordered or added, and a summary
that disagrees with `git log` is worse than no summary.

Every field here is read from git. The script invents nothing: subjects, file
names, insertion and deletion counts and the body's first paragraph all come
from `git log` and `git show --numstat`.

Run after committing:

    python tools/gen_commit_history.py

By default it documents the commits not yet in `origin/master`. Pass an
explicit range to document something else:

    python tools/gen_commit_history.py origin/master..HEAD
    python tools/gen_commit_history.py v1.0..v1.1
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "COMMITS.md"
DEFAULT_RANGE = "origin/master..HEAD"

# A commit touching more files than this gets its file list summarised by
# directory instead of listed in full, so one sweeping commit cannot bury the
# rest of the history under a wall of paths.
MAX_FILES_LISTED = 14

SEP = "\x1e"  # record separator, safe inside commit messages


def git(*args: str) -> str:
    """Run git in the repository and return stdout, raising on failure."""
    result = subprocess.run(
        ("git", *args), cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def classify(subject: str, body: str) -> str:
    """Label a commit from its own words.

    Deliberately conservative: anything that does not clearly announce itself
    as a fix, a document or an addition is left uncategorised rather than
    guessed at.
    """
    text = f"{subject}\n{body}".lower()
    if re.search(r"\b(fix|correct|stop|do not let|repair)\b", subject.lower()):
        return "fix"
    if re.search(r"\b(document|record|consolidate)\b", subject.lower()):
        return "docs"
    if re.search(r"\b(add|expose|introduce|bring|make)\b", subject.lower()):
        return "feature"
    return ""


def first_paragraph(body: str, limit: int = 320) -> str:
    """The commit message's opening paragraph, collapsed to one line."""
    body = body.strip()
    if not body:
        return ""
    para = re.split(r"\n\s*\n", body)[0]
    para = re.sub(r"\s+", " ", para).strip()
    if len(para) > limit:
        cut = para[:limit].rsplit(" ", 1)[0]
        para = cut.rstrip(",.;:") + "…"
    return para


def summarise_dirs(files: list[tuple[str, int, int]]) -> list[str]:
    """Collapse a long file list to per-directory counts."""
    buckets: dict[str, list[int]] = {}
    for path, add, rem in files:
        key = str(Path(path).parent) if "/" in path else "(root)"
        slot = buckets.setdefault(key, [0, 0, 0])
        slot[0] += 1
        slot[1] += add
        slot[2] += rem
    lines = []
    for key in sorted(buckets):
        count, add, rem = buckets[key]
        noun = "file" if count == 1 else "files"
        lines.append(f"`{key}/` — {count} {noun}, +{add} −{rem}")
    return lines


def collect(rev_range: str) -> list[dict]:
    """Read every commit in the range, with its files and line counts."""
    fmt = SEP.join(["%H", "%h", "%an", "%ad", "%s", "%b"]) + "\x1d"
    raw = git("log", "--reverse", "--date=short", f"--format={fmt}", rev_range)
    commits = []
    for record in raw.split("\x1d"):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(SEP)
        if len(parts) < 6:
            continue
        full, short, author, date, subject, body = parts[:6]

        numstat = git("show", "--numstat", "--format=", full)
        files: list[tuple[str, int, int]] = []
        for line in numstat.splitlines():
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            add_s, rem_s, path = cols
            # Binary files report "-" instead of a count.
            add = int(add_s) if add_s.isdigit() else 0
            rem = int(rem_s) if rem_s.isdigit() else 0
            files.append((path, add, rem))

        commits.append(
            {
                "full": full,
                "short": short,
                "author": author,
                "date": date,
                "subject": subject,
                "body": body,
                "files": files,
                "adds": sum(f[1] for f in files),
                "rems": sum(f[2] for f in files),
                "kind": classify(subject, body),
            }
        )
    return commits


def render(commits: list[dict], rev_range: str) -> str:
    total_files = len({f[0] for c in commits for f in c["files"]})
    total_adds = sum(c["adds"] for c in commits)
    total_rems = sum(c["rems"] for c in commits)

    out: list[str] = []
    out.append("# Commit history")
    out.append("")
    out.append(
        "Generated by `tools/gen_commit_history.py` from `git log`. Do not edit "
        "by hand; rerun the script after committing."
    )
    out.append("")
    out.append(
        f"Range `{rev_range}`: {len(commits)} commits, {total_files} files "
        f"touched, +{total_adds} −{total_rems} lines."
    )
    out.append("")
    out.append(
        "Line counts are summed per commit, so they measure churn rather than "
        "the net difference against the base. A file edited by several commits "
        "contributes each time, and a line added then removed contributes to "
        "both totals. `git diff --shortstat` against the base reports a smaller "
        "figure for the same range; both are correct and answer different "
        "questions."
    )
    out.append("")

    # Summary table first, so the history can be scanned without scrolling.
    out.append("## Summary")
    out.append("")
    out.append("| # | commit | date | subject | files | +/− |")
    out.append("|---|---|---|---|---:|---|")
    for i, c in enumerate(commits, 1):
        subject = c["subject"].replace("|", "\\|")
        out.append(
            f"| {i} | `{c['short']}` | {c['date']} | {subject} | "
            f"{len(c['files'])} | +{c['adds']} −{c['rems']} |"
        )
    out.append("")

    out.append("## Commits")
    out.append("")
    for i, c in enumerate(commits, 1):
        kind = f" _{c['kind']}_" if c["kind"] else ""
        out.append(f"### {i}. {c['subject']}")
        out.append("")
        out.append(
            f"`{c['short']}` · {c['date']} · {c['author']} · "
            f"{len(c['files'])} files, +{c['adds']} −{c['rems']}{kind}"
        )
        out.append("")
        summary = first_paragraph(c["body"])
        if summary:
            out.append(summary)
            out.append("")

        files = sorted(c["files"], key=lambda f: -(f[1] + f[2]))
        if len(files) > MAX_FILES_LISTED:
            out.append(
                f"Files by directory ({len(files)} files, listed individually "
                f"only below {MAX_FILES_LISTED + 1}):"
            )
            out.append("")
            for line in summarise_dirs(files):
                out.append(f"- {line}")
        else:
            out.append("Files:")
            out.append("")
            for path, add, rem in files:
                out.append(f"- `{path}` — +{add} −{rem}")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    rev_range = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RANGE
    commits = collect(rev_range)
    if not commits:
        print(f"no commits in range {rev_range}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(commits, rev_range), encoding="utf-8")
    print(f"wrote {OUT}: {len(commits)} commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
