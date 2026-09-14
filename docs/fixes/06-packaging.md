# Fix 6 — Packaging and dependency metadata

**Status:** fixed; Apache-2.0 metadata in place, copyright holder still unset

**Severity:** low — blocks clean installation and misstates the license
**Files:** `requirements.txt`, `pyproject.toml`, `pytest.ini` (new)

## Defects

**`requirements.txt` was unusable.** It pinned a set of transitive
dependencies while omitting both packages the benchmark cannot run without:

```
antlr4-python3-runtime==4.9.3
dataclasses==0.8          <-- Python 3.6 backport; breaks install on 3.7+
hydra-core==1.3.2
importlib-resources==5.4.0
omegaconf==2.3.0
packaging==21.3
pyparsing==3.1.4
PyYAML==6.0.1
zipp==3.6.0
```

`torch` and `mpi4py` were absent. `dataclasses==0.8` is a backport of a module
that entered the standard library in Python 3.7; installing it on any modern
interpreter fails or shadows the stdlib module.

**License metadata contradicted the README.** `pyproject.toml` declared
`License :: OSI Approved :: MIT License`; `README.md` states Apache 2.0 plus
the DOE government-rights notice. No `LICENSE` file existed and the
`license = { file = "LICENSE" }` line was commented out.

**`requires-python = ">=3.8"`** was inaccurate — Python 3.8 is end-of-life and
the codebase uses `X | Y` type syntax requiring 3.10 at runtime in annotated
positions.

## Changes

`requirements.txt` now lists direct dependencies only, unpinned to minor
versions, with `torch` and `mpi4py` documented but commented out because on
Aurora and other ALCF systems they come from `module load frameworks` rather
than from an index.

`pyproject.toml`:

- classifier changed to `License :: OSI Approved :: Apache Software License`
  to match the README;
- `requires-python` raised to `>=3.9`;
- the 3.8 classifier removed.

`pytest.ini` registers the `gloo` marker and sets `testpaths`, so
`python -m pytest` works from the repository root and
`-m "not gloo"` selects the fast unit subset.

## Not done

**No `LICENSE` file is committed.** The README names Apache 2.0 and carries a
DOE government-rights notice, which implies a specific copyright holder line —
UChicago Argonne, LLC, in the usual ALCF form. Choosing that text is a legal
decision for the project, not a mechanical fix. Add the file and re-enable the
commented `license = { file = "LICENSE" }` line in `pyproject.toml`.

**`tests/test.sh` previously hardcoded another user's `datascience_collab` paths** and could not
run for anyone outside that project. It is left untouched; the new suite under
`tests/` supersedes it and runs anywhere.
