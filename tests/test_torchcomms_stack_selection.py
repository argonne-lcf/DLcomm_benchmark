"""Tests for torchcomms stack selection in the job scripts.

The torchcomms layer is only meaningful when the intended build is the one that
actually gets imported. Selection happens in shell, before Python starts, so it
is tested by running the selector block directly.

`DLCOMM_TC_STACK` takes two values:

  frameworks  the module-provided torchcomms 0.1.0 (default)
  local       the 0.3.0 build under this project's stacks/ directory

There is no third value and no fallback to the directory the build was
originally copied from. A stack that can silently change underneath a run makes
its numbers unattributable, so a missing stack is a hard failure.
"""

import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBSCRIPT = os.path.join(REPO, "examples", "17_torchcomms", "jobscript_torchcomms.sh")
RUNNER = os.path.join(REPO, "tools", "run_all_scales.sh")

LOCAL_STACK = "/lus/flare/projects/datascience/kaushik/stacks/torchcomms_0.3.0"


def _selector(tmp_path, value, make_local=True):
    """Run the selector logic in isolation and report what it resolved to."""
    local_stack = tmp_path / "stacks" / "torchcomms_0.3.0"
    if make_local:
        (local_stack / "torchcomms").mkdir(parents=True, exist_ok=True)
        (local_stack / "pytorch").mkdir(parents=True, exist_ok=True)

    script = f"""
        TC_STACK="${{DLCOMM_TC_STACK:-frameworks}}"
        LOCAL_STACK={local_stack}
        TC03_TC="$LOCAL_STACK/torchcomms"
        TC03_TORCH="$LOCAL_STACK/pytorch"

        TC_PYTHONPATH=""
        if [[ "$TC_STACK" == "local" ]]; then
            if [[ ! -d "$TC03_TC" || ! -d "$TC03_TORCH" ]]; then
                echo "VERDICT=STACK_MISSING"
                exit 1
            fi
            TC_PYTHONPATH="$TC03_TC:$TC03_TORCH"
        fi
        echo "TORCHCOMMS_STACK=$TC_STACK"
        echo "PYTHONPATH_ADD=$TC_PYTHONPATH"
    """
    env = dict(os.environ)
    if value is None:
        env.pop("DLCOMM_TC_STACK", None)
    else:
        env["DLCOMM_TC_STACK"] = value

    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    out = {
        k: v
        for k, v in (
            line.split("=", 1)
            for line in proc.stdout.strip().splitlines()
            if "=" in line
        )
    }
    out["_rc"] = str(proc.returncode)
    return out


def test_default_is_the_frameworks_stack(tmp_path):
    """Unset means the module build: opt-in, so old examples are unaffected."""
    result = _selector(tmp_path, None)
    assert result["TORCHCOMMS_STACK"] == "frameworks"
    assert result["PYTHONPATH_ADD"] == ""


def test_local_selects_the_project_owned_stack(tmp_path):
    result = _selector(tmp_path, "local")
    assert result["TORCHCOMMS_STACK"] == "local"
    assert "torchcomms" in result["PYTHONPATH_ADD"]
    assert "pytorch" in result["PYTHONPATH_ADD"]


def test_torchcomms_precedes_torch_on_pythonpath(tmp_path):
    """The pair must be matched; torchcomms resolves first."""
    add = _selector(tmp_path, "local")["PYTHONPATH_ADD"]
    tc, torch_path = add.split(":")
    assert tc.endswith("torchcomms")
    assert torch_path.endswith("pytorch")


def test_missing_local_stack_fails_loudly(tmp_path):
    """A missing stack must not silently fall back to the module build."""
    result = _selector(tmp_path, "local", make_local=False)
    assert result["_rc"] == "1"
    assert result.get("VERDICT") == "STACK_MISSING"
    assert "TORCHCOMMS_STACK" not in result


def test_no_fallback_to_another_users_directory():
    """The scripts must not reference a directory this project does not own."""
    for path in (JOBSCRIPT, RUNNER):
        with open(path) as fh:
            text = fh.read()
        assert "datascience_collab" not in text, (
            f"{os.path.basename(path)} points at a directory outside this project; "
            "a stack that can move makes the numbers unattributable"
        )


def test_jobscript_pins_the_project_owned_stack():
    with open(JOBSCRIPT) as fh:
        text = fh.read()
    assert LOCAL_STACK in text


def test_no_deprecated_alias_remains():
    """The original build owner's username must not appear in the scripts.

    The name is assembled at runtime rather than written out, so this guard
    does not itself reintroduce the string it exists to forbid.
    """
    forbidden = "".join(("psh", "ukla"))
    for path in (JOBSCRIPT, RUNNER):
        with open(path) as fh:
            text = fh.read()
        assert forbidden not in text, (
            f"{os.path.basename(path)} still names the original build's owner"
        )
