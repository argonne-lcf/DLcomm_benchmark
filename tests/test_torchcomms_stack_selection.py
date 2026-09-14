"""Tests for torchcomms stack selection in the job scripts.

The torchcomms layer is only meaningful when the intended build is the one that
actually gets imported. Selection happens in shell, before Python starts, so
these tests drive the shell logic directly rather than importing anything.

Two failures motivate them:

1. The 0.3.0 build originally lived in another user's directory. The selector
   value was named after that user (`pshukla`). Renaming it to `local` risks
   silently breaking submissions that still pass the old value, so the alias is
   tested rather than assumed.
2. A missing stack must not fall through to the shipped 0.1.0 build, which
   answers `all_reduce` and refuses everything else. A run that quietly
   measures the wrong stack looks like a successful run.
"""

import subprocess
import textwrap

import pytest


REPO_ROOT_MARKER = "examples/17_torchcomms/jobscript_torchcomms.sh"


def _selector(tmp_path, stack_value, make_local=True):
    """Run the stack-selection block and return its printed keys.

    The block is reproduced here rather than sourced: the real job script runs
    `module load frameworks` and `qsub`-only logic at import time, neither of
    which exists off Aurora. Keep this in sync with the job scripts.
    """
    local_stack = tmp_path / "stack"
    if make_local:
        (local_stack / "torchcomms").mkdir(parents=True, exist_ok=True)
        (local_stack / "pytorch").mkdir(parents=True, exist_ok=True)

    script = textwrap.dedent(
        f"""
        TC_STACK="${{DLCOMM_TC_STACK:-frameworks}}"
        if [[ "$TC_STACK" == "pshukla" ]]; then
            echo "TC_STACK_NOTE=pshukla is a deprecated alias for local"
            TC_STACK=local
        fi
        LOCAL_STACK={local_stack}
        if [[ -d "$LOCAL_STACK/torchcomms" && -d "$LOCAL_STACK/pytorch" ]]; then
            TC03_TC="$LOCAL_STACK/torchcomms"
            TC03_TORCH="$LOCAL_STACK/pytorch"
        else
            echo "TC_STACK_WARN=local copy missing, falling back"
            TC03_TORCH=/upstream/pytorch
            TC03_TC=/upstream/torchcomms
        fi
        TC_PYTHONPATH=""
        if [[ "$TC_STACK" == "local" ]]; then
            TC_PYTHONPATH="$TC03_TC:$TC03_TORCH"
        fi
        echo "TORCHCOMMS_STACK=$TC_STACK"
        echo "PYTHONPATH_ADD=$TC_PYTHONPATH"
        """
    )
    env = {"PATH": "/usr/bin:/bin"}
    if stack_value is not None:
        env["DLCOMM_TC_STACK"] = stack_value
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=True
    ).stdout
    return dict(
        line.split("=", 1) for line in out.strip().splitlines() if "=" in line
    )


def test_default_is_frameworks(tmp_path):
    """No selection means the shipped build, not the custom one."""
    result = _selector(tmp_path, None)
    assert result["TORCHCOMMS_STACK"] == "frameworks"
    assert result["PYTHONPATH_ADD"] == ""


def test_local_selects_the_project_copy(tmp_path):
    result = _selector(tmp_path, "local")
    assert result["TORCHCOMMS_STACK"] == "local"
    assert result["PYTHONPATH_ADD"].endswith("stack/torchcomms:" + str(tmp_path / "stack" / "pytorch"))


def test_pshukla_is_a_deprecated_alias_for_local(tmp_path):
    """The old value must keep working, and say so."""
    result = _selector(tmp_path, "pshukla")
    assert result["TORCHCOMMS_STACK"] == "local"
    assert "deprecated alias" in result["TC_STACK_NOTE"]


def test_alias_and_local_resolve_identically(tmp_path):
    """The alias must not be a second, subtly different code path."""
    assert (
        _selector(tmp_path, "pshukla")["PYTHONPATH_ADD"]
        == _selector(tmp_path, "local")["PYTHONPATH_ADD"]
    )


def test_missing_copy_warns_rather_than_selecting_silently(tmp_path):
    """A missing stack must be audible in the log, not a silent fallback."""
    result = _selector(tmp_path, "local", make_local=False)
    assert "TC_STACK_WARN" in result
    assert result["PYTHONPATH_ADD"].startswith("/upstream/")


@pytest.mark.parametrize("name", ["16_all_layers_comparison", "17_torchcomms"])
def test_jobscripts_document_the_current_selector(name):
    """Job scripts must not advertise the retired value as the way to opt in."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    scripts = list((root / "examples" / name).glob("jobscript*.sh"))
    assert scripts, f"no jobscript found for {name}"
    for script in scripts:
        text = script.read_text()
        assert "DLCOMM_TC_STACK=local" in text, f"{script} lost the current selector"
        assert "-v DLCOMM_TC_STACK=pshukla" not in text, (
            f"{script} still tells the user to submit with the retired value"
        )
