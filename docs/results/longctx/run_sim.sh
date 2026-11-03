#!/bin/sh
# 4M at work: deepinfer-server with a 4,259,840-token context, then a simulated user loads this
# repository and the Python standard library through its HTTP API and asks questions as it goes.
# The server stays up afterwards.  Its key is made here and only passed on the command line.
# Usage: run_sim.sh NAME [server options, e.g. --check-copies --self-check]
set -u
NAME=$1; shift
cd /root/autodl-tmp/di-4m || exit 1
mkdir -p logs
KEY=$(/root/miniconda3/bin/python -c 'import secrets; print(secrets.token_urlsafe(12))')
/root/miniconda3/bin/python /root/autodl-tmp/di-claude-c/drop_cache.py
(
    while true; do
        printf '%s current %s %s gpu %s\n' "$(date +%T)" "$(cat /sys/fs/cgroup/memory.current)" \
            "$(grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat | tr '\n' ' ')" \
            "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
        sleep 30
    done
) > "logs/${NAME}_mem.log" 2>&1 &
MONITOR=$!
nohup /root/miniconda3/bin/python apps/longctx_chat.py --capacity 4259840 --key "$KEY" "$@" \
    > "logs/${NAME}_server.log" 2>&1 < /dev/null &
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:6006/status?key=$KEY" > /dev/null && break
    sleep 5
done
/root/miniconda3/bin/python scripts/longctx_user_sim.py run --url http://127.0.0.1:6006 --key "$KEY" \
    --project /root/autodl-tmp/di-4m --out "logs/$NAME.json" > "logs/$NAME.log" 2>&1
echo "exit $?" >> "logs/$NAME.log"
kill "$MONITOR"
