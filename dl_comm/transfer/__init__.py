"""Host-to-device transfer measurement for DLcomm.

Collectives run device-to-device. A degraded PCIe path -- a rank on a remote
NUMA node, a link trained below full width, pageable host staging -- slows real
training while leaving collective benchmarks looking healthy. Measuring H2D and
D2H directly turns that invisible ceiling into a reported number.
"""

from dl_comm.transfer.h2d import (
    TransferResult,
    fill_shuffled,
    format_table,
    measure,
)

__all__ = ["TransferResult", "fill_shuffled", "format_table", "measure"]
