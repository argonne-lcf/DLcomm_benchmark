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

for required in (MB, RAIL):
    if not required.is_file():
        sys.exit(f"FATAL: expected file not found: {required}")

# Both edits are validated before either is written. Writing meson.build and
# then failing on libfabric_rail.cpp leaves a half-patched tree that builds
# with the probe defined but the guard still on the version test -- a state
# neither the patched nor unpatched build expects, and one a re-run cannot
# detect because the first fix reports "already patched".
pending = []   # (path, new_text)

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
    pending.append((MB, mb.replace(anchor, probe, 1)))

# ---- Fix 2: use the probe macro ----------------------------------------
rail = RAIL.read_text()
old = "#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)\n                mr_attr.device.rocr = device_id;\n#endif"
new = "#ifdef NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR\n                mr_attr.device.rocr = device_id;\n#endif"
if new in rail:
    print("libfabric_rail.cpp: already patched")
elif old in rail:
    pending.append((RAIL, rail.replace(old, new, 1)))
else:
    # Nothing has been written yet, so the tree is still pristine.
    sys.exit("FATAL: libfabric_rail.cpp rocr guard not found in expected form")

# ---- commit: every edit validated, now apply ----------------------------
changed = []
written = []
try:
    for path, text in pending:
        backup = path.with_suffix(path.suffix + ".prepatch")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(text)
        written.append((path, backup))
        changed.append(str(path))
except Exception as exc:                                  # pragma: no cover
    for path, backup in written:
        try:
            path.write_text(backup.read_text())
        except OSError:
            print(f"  WARNING: could not roll back {path}", file=sys.stderr)
    sys.exit(f"FATAL: patch failed, rolled back {len(written)} file(s): {exc}")

print("PATCHED:" if changed else "NO CHANGES")
for c in changed:
    print("  ", c)
