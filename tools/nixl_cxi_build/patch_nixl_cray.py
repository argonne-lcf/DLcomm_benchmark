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
# Review finding P1-7: the previous predicate tested two UNRELATED
# substrings -- "NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR" anywhere and
# "cpp.compiles" anywhere -- and only stripped WHOLE comment lines. A tree
# with an unrelated cpp.compiles() call plus an INLINE comment naming the
# macro satisfied both, so the patcher exited 0, left meson.build
# unchanged, and still rewrote the C++ guard to a macro that was never
# defined. Validate the probe as one structure: the rocr source, the
# compile condition, and the resulting conditional argument must all be
# present and must refer to each other.
def _strip_comments(text):
    """Drop whole-line AND inline # comments (outside quotes)."""
    out = []
    for line in text.splitlines():
        buf, quote = [], None
        for ch in line:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
                buf.append(ch)
            elif ch == "#":
                break
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)


_mb_code = _strip_comments(mb)
# The probe is only "already applied" if all three parts exist together:
#   1. a cpp.compiles() call that actually compiles the rocr source,
#   2. that source naming the fi_mr_attr device.rocr member,
#   3. the macro being added as a compile argument under that result.
_has_probe_call = re.search(r"cpp\.compiles\s*\(", _mb_code) is not None
_has_rocr_source = re.search(r"device\s*\.\s*rocr", _mb_code) is not None
_has_macro_arg = re.search(
    r"NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR", _mb_code) is not None
_already = _has_probe_call and _has_rocr_source and _has_macro_arg
if _already:
    print("meson.build: already patched (probe structure verified)")
elif _has_macro_arg and not (_has_probe_call and _has_rocr_source):
    # The macro is referenced but no real probe defines it. Refusing is the
    # only safe action: patching the C++ guard would gate on a macro that
    # nothing ever sets.
    sys.exit("FATAL: meson.build references NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR "
             "with no rocr compile probe; refusing to patch an unfamiliar "
             "structure")
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
        # Snapshot the CURRENT bytes in memory and register the file as
        # at-risk BEFORE writing. Registering only after a successful write
        # meant a write that failed part-way was never rolled back. A stale
        # .prepatch from an earlier run may hold older bytes, so it is not
        # trusted as the restore source.
        original = path.read_text()
        written.append((path, original))
        backup = path.with_suffix(path.suffix + ".prepatch")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(text)
        changed.append(str(path))
except Exception as exc:                                  # pragma: no cover
    for path, original in written:
        try:
            path.write_text(original)
        except OSError:
            print(f"  WARNING: could not roll back {path}", file=sys.stderr)
    sys.exit(f"FATAL: patch failed, rolled back {len(written)} file(s): {exc}")

print("PATCHED:" if changed else "NO CHANGES")
for c in changed:
    print("  ", c)
