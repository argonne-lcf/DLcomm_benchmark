"""Shared pytest fixtures and a real multi-process gloo harness.

The tests in this directory fall into two kinds:

* pure unit tests, which import DLcomm modules and check arithmetic;
* **integration tests**, which spawn real ``torch.distributed`` processes over
  the CPU ``gloo`` backend and run actual collectives.

The gloo harness needs no GPU, no MPI launcher, and no allocation, so the whole
suite runs on a login node or in CI in well under a minute.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "gloo: spawns real torch.distributed processes")


@pytest.fixture(scope="session")
def torch_mod():
    torch = pytest.importorskip("torch", reason="torch is required")
    return torch


@pytest.fixture
def fake_logger():
    class _Logger:
        def __init__(self):
            self.lines = []

        def output(self, msg=""):
            self.lines.append(str(msg))

        info = output
        warning = output
        error = output

        def text(self):
            return "\n".join(self.lines)

    return _Logger()


class FakeModeCfg:
    """Stand-in for the omegaconf node describing one communication mode."""

    def __init__(self, num_compute_nodes, device_ids_per_node,
                 num_devices_per_node=None):
        self.num_compute_nodes = num_compute_nodes
        self.device_ids_per_node = list(device_ids_per_node)
        self.num_devices_per_node = (num_devices_per_node
                                     if num_devices_per_node is not None
                                     else len(device_ids_per_node))


@pytest.fixture
def mode_cfg_factory():
    return FakeModeCfg
