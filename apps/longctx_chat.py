"""The long-context chat server's earlier entry point, kept for the command lines that use it.

The server is now :mod:`deepinfer.cli.serve` - ``deepinfer-server`` once the package is
installed.  This runs it with the defaults the AutoDL runs were made with: every address,
port 6006 (what the console's custom-service link opens), a 3,407,872-token context and
the coder model under ``--models-dir``, its config and tokenizer under ``--meta-root``:

    python apps/longctx_chat.py --port 6006 --capacity 3407872

Any other option is the server's (``python apps/longctx_chat.py --help``) and wins over
these defaults.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.cli import main as cli


def server_argv(argv: list[str]) -> list[str]:
    """``deepinfer server`` arguments for this script's command line."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    paths, rest = parser.parse_known_args(argv)
    defaults = [
        "--host", "0.0.0.0", "--port", "6006", "--ctx-size", "3407872", "--encode-window", "98304",
        "-m", os.path.join(paths.models_dir, "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"),
        "--meta", os.path.join(paths.meta_root, "meta_coder"),
    ]
    return ["server", *defaults, *rest]  # given later, the caller's options win


def main() -> None:
    cli.main(server_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
