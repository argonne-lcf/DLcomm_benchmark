# shellcheck shell=bash
#
# Shared PBS nodefile helpers.
#
# Source this from a jobscript:
#
#     source "$(dirname "$0")/../../tools/nodefile.sh"
#     NNODES=$(dlcomm_node_count)
#
# WHY THIS EXISTS
#
# `wc -l < $PBS_NODEFILE` counts *newlines*, not lines. On Tara the PBS
# nodefile has no trailing newline on its last entry, so a 1-node allocation
# reports 0 and the jobscript launches `mpiexec --np 0`, which dies with:
#
#     Invalid -n/--np value 0: outside of valid range 1-2147483647
#
# Measured on tara-n (north-asn-01), nodefile from a 1-node job:
#
#     wc -l  => 0
#     grep -c . => 1
#     bytes: 45       last byte: 0x76 ('v')   # no 0x0a
#
# `grep -c .` counts non-empty lines whether or not the file ends in a
# newline, which is the count we actually want. Every jobscript must use
# dlcomm_node_count instead of `wc -l`.

# Count the nodes in $PBS_NODEFILE (or the file given as $1).
# Prints 1 when no nodefile is set, so interactive/non-PBS runs still work.
dlcomm_node_count() {
    local nf="${1:-${PBS_NODEFILE:-}}"
    if [ -z "$nf" ] || [ ! -r "$nf" ]; then
        echo 1
        return 0
    fi
    local n
    n=$(grep -c . "$nf" 2>/dev/null || echo 0)
    if [ "$n" -lt 1 ]; then
        echo 1
    else
        echo "$n"
    fi
}

# Abort loudly if a computed rank/node count is unusable. Launching
# `mpiexec --np 0` wastes a whole allocation and the error surfaces far from
# the cause, so fail here instead.
dlcomm_require_positive() {
    local name="$1" val="$2"
    case "$val" in
        ''|*[!0-9]*)
            echo "FATAL: $name is not a positive integer (got '${val}')" >&2
            exit 1
            ;;
    esac
    if [ "$val" -lt 1 ]; then
        echo "FATAL: $name must be >= 1 (got ${val})" >&2
        exit 1
    fi
}
