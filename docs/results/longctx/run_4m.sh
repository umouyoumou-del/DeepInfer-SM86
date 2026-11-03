#!/bin/sh
# The 39-needle suite at 4M (4,194,304 tokens) with the 3M 21/21 run's settings (experts in VRAM,
# near window 96K, everything else default).  Logs the container's memory and the GPU every 30 s.
# Usage: run_4m.sh NAME [extra needle_complex.py args]
set -u
NAME=$1; shift
cd /root/autodl-tmp/di-4m || exit 1
mkdir -p logs
/root/miniconda3/bin/python /root/autodl-tmp/di-claude-c/drop_cache.py
(
    while true; do
        printf '%s current %s peak %s %s gpu %s\n' "$(date +%T)" "$(cat /sys/fs/cgroup/memory.current)" \
            "$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo -)" \
            "$(grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat | tr '\n' ' ')" \
            "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
        sleep 30
    done
) > "logs/${NAME}_mem.log" 2>&1 &
MONITOR=$!
/root/miniconda3/bin/python scripts/needle_complex.py --tokens 4194304 --unpin no --near-window 98304 \
    --out "logs/$NAME.json" "$@" > "logs/$NAME.log" 2>&1
echo "exit $?" >> "logs/$NAME.log"
kill "$MONITOR"
