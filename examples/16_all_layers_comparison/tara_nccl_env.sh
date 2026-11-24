#!/bin/bash
# Source explicitly for the validated Tara GH200 + Libfabric/CXI stack.
# Never silently change global NCCL defaults on other platforms.
case "${NCCL_PROTO:-Simple}" in
  Simple|LL) export NCCL_PROTO=${NCCL_PROTO:-Simple} ;;
  *) printf '%s\n' 'Tara profile refuses this NCCL_PROTO: LL128 corrupted allreduce data in two-node validation. Use Simple or LL; vendor root cause remains open.' >&2; return 2 2>/dev/null || exit 2 ;;
esac
printf 'Tara NCCL profile: NCCL_PROTO=%s; LL128 excluded due to observed corruption (not a vendor fix).\n' "$NCCL_PROTO" >&2
