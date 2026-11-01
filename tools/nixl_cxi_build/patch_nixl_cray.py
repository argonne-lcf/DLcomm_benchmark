#!/usr/bin/env python3
"""Patch NIXL for Cray libfabric 2.3.1 on tara-n.

Fix 1: src/utils/libfabric/meson.build  -- add a compile probe for the
       fi_mr_attr.device.rocr union member (version test is wrong predicate).
Fix 2: src/utils/libfabric/libfabric_rail.cpp -- gate the rocr assignment on
       the new probe macro instead of FI_VERSION(2,3).
Idempotent: re-running makes no further changes.
"""
import re, sys, pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MB = ROOT / "src/utils/libfabric/meson.build"
RAIL = ROOT / "src/utils/libfabric/libfabric_rail.cpp"

changed = []

# ---- Fix 1: meson probe -------------------------------------------------
mb = MB.read_text()
if "NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR" in mb:
    print("meson.build: already patched")
else:
    anchor = "        message('libfabric does not support FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')\n    endif\n"
    if anchor not in mb:
        sys.exit("FATAL: meson.build anchor not found")
    probe = """        message('libfabric does not support FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')
    endif

    # Probe for the fi_mr_attr.device.rocr union member. NIXL guards its use
    # with FI_VERSION(FI_MAJOR,FI_MINOR) >= FI_VERSION(2,3), but Cray
    # libfabric 2.3.1 reports API 2.3 and has NO rocr member in the device
    # union (it landed later upstream). A version test is the wrong
    # predicate here; probe the member itself.
    rocr_probe_src = '''#include <rdma/fi_domain.h>
int main(void) { struct fi_mr_attr a; a.device.rocr = 0; return 0; }
'''
    if cpp.compiles(rocr_probe_src,
                    name: 'fi_mr_attr.device.rocr exists',
                    dependencies: libfabric_dep)
        libfabric_utils_cpp_args += ['-DNIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR']
        message('libfabric has fi_mr_attr.device.rocr')
    else
        message('libfabric lacks fi_mr_attr.device.rocr - ROCr device id not set')
    endif
"""
    mb = mb.replace(anchor, probe, 1)
    MB.write_text(mb)
    changed.append(str(MB))

# ---- Fix 2: use the probe macro ----------------------------------------
rail = RAIL.read_text()
old = "#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)\n                mr_attr.device.rocr = device_id;\n#endif"
new = "#ifdef NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR\n                mr_attr.device.rocr = device_id;\n#endif"
if new in rail:
    print("libfabric_rail.cpp: already patched")
elif old in rail:
    RAIL.write_text(rail.replace(old, new, 1))
    changed.append(str(RAIL))
else:
    sys.exit("FATAL: libfabric_rail.cpp rocr guard not found in expected form")

print("PATCHED:" if changed else "NO CHANGES")
for c in changed:
    print("  ", c)
