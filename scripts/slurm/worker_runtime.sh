#!/bin/bash
# Source this before Python starts. AF_UNIX socket paths are limited to ~108
# bytes; deeply nested experiment paths cannot be used as multiprocessing TMPDIR.
# Scientific outputs remain in the checkout; /tmp holds transient IPC files only.
DD_IPC_TMP=$(mktemp -d "/tmp/dd-ipc.${SLURM_JOB_ID:-local}.XXXXXX")
export TMPDIR="$DD_IPC_TMP"
trap 'rm -rf -- "$DD_IPC_TMP"' EXIT
