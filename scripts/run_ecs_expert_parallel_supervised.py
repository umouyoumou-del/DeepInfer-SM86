"""Supervise the two-rank NCCL expert-parallel benchmark.

The supervisor starts a remote-expert rank and a coordinator rank, monitors
process exit and the remote heartbeat file, and retries the whole request after
restarting both ranks.  A fault-injection switch is provided for recovery tests.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _heartbeat_age(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - float(path.read_text(encoding="ascii").strip()))
    except Exception:
        return None


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _parse_output(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF"
            "/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--meta-dir", type=Path, default=Path("/root/autodl-tmp/meta"))
    parser.add_argument("--prompt", default="只回答数字：5-2=?")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--remote-fraction", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--worker-fault-after",
        type=int,
        default=0,
        help="first-attempt remote request count before simulated rank1 exit",
    )
    parser.add_argument("--heartbeat-timeout", type=float, default=12.0)
    parser.add_argument("--attempt-timeout", type=float, default=300.0)
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=Path("/root/autodl-tmp/results/nccl_supervised_heartbeat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/results/ecs_expert_parallel_supervised.json"),
    )
    args = parser.parse_args()

    script = Path(__file__).with_name("bench_ecs_expert_parallel_nccl.py")
    repo = Path(__file__).resolve().parent.parent
    log_dir = args.output.parent / (args.output.stem + "_attempts")
    log_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0
    last_worker_rc: int | None = None
    last_coord_rc: int | None = None
    for attempt in range(max(1, int(args.max_retries) + 1)):
        attempts = attempt + 1
        port = _free_port()
        heartbeat = args.heartbeat_file.with_name(
            f"{args.heartbeat_file.stem}.{attempt}{args.heartbeat_file.suffix}"
        )
        try:
            heartbeat.unlink()
        except FileNotFoundError:
            pass
        common = [
            sys.executable,
            str(script),
            "--model",
            str(args.model),
            "--meta-dir",
            str(args.meta_dir),
            "--prompt",
            args.prompt,
            "--requests",
            str(args.requests),
            "--concurrent",
            str(args.concurrent),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-cache",
            str(args.max_cache),
            "--remote-fraction",
            str(args.remote_fraction),
            "--heartbeat-file",
            str(heartbeat),
        ]
        base_env = os.environ.copy()
        base_env.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": "2",
                "PYTHONPATH": str(repo),
            }
        )
        worker_env = dict(base_env)
        worker_env.update({"RANK": "1", "LOCAL_RANK": "1"})
        worker_env["DEEPINFER_REMOTE_FAULT_AFTER"] = (
            str(args.worker_fault_after) if attempt == 0 else "0"
        )
        coord_env = dict(base_env)
        coord_env.update({"RANK": "0", "LOCAL_RANK": "0"})
        worker_log = open(log_dir / f"attempt{attempt}.worker.log", "wb")
        coord_log = open(log_dir / f"attempt{attempt}.coord.log", "wb")
        worker: subprocess.Popen | None = None
        coord: subprocess.Popen | None = None
        try:
            worker = subprocess.Popen(
                common,
                env=worker_env,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(0.5)
            coord_cmd = list(common) + ["--output", str(args.output)]
            coord = subprocess.Popen(
                coord_cmd,
                env=coord_env,
                stdout=coord_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.perf_counter()
            failure: str | None = None
            while True:
                last_worker_rc = worker.poll()
                last_coord_rc = coord.poll()
                if last_worker_rc is not None and last_worker_rc != 0:
                    failure = f"worker_exit_{last_worker_rc}"
                    break
                if last_coord_rc is not None:
                    if last_coord_rc == 0 and _parse_output(args.output) is not None:
                        result = _parse_output(args.output)
                        report = {
                            "recovered": attempt > 0,
                            "attempts": attempts,
                            "worker_rc": last_worker_rc,
                            "coordinator_rc": last_coord_rc,
                            "result": result,
                            "log_dir": str(log_dir),
                        }
                        args.output.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(json.dumps(report, ensure_ascii=False, indent=2))
                        return 0
                    failure = f"coordinator_exit_{last_coord_rc}"
                    break
                age = _heartbeat_age(heartbeat)
                if (
                    age is not None
                    and age > float(args.heartbeat_timeout)
                ):
                    failure = f"heartbeat_stale_{age:.1f}s"
                    break
                if time.perf_counter() - started > float(args.attempt_timeout):
                    failure = "attempt_timeout"
                    break
                time.sleep(0.5)
            print(
                f"[supervisor] attempt {attempt} failed: {failure}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            _terminate(worker)
            _terminate(coord)
            worker_log.close()
            coord_log.close()
    print(
        json.dumps(
            {
                "recovered": False,
                "attempts": attempts,
                "worker_rc": last_worker_rc,
                "coordinator_rc": last_coord_rc,
                "log_dir": str(log_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
