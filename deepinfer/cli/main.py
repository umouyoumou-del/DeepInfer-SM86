"""The ``deepinfer`` command and its llama.cpp-style aliases.

    deepinfer-cli    -hf coder              chat in the terminal (deepinfer cli)
    deepinfer-server -hf coder -c 1048576   page + API over a long context (deepinfer server)
    deepinfer-bench  -m model.gguf          prompt / generation speed (deepinfer bench)
    deepinfer pull   -hf chat               download a model and its config / tokenizer
    deepinfer info                          what this machine offers (GPU, CUDA, Triton, CuPy)

A model is a local GGUF (``-m``; ``--meta`` for its config/tokenizer folder) or a Hugging Face
repo (``-hf org/repo[:QUANT]``, or a preset: ``coder``, ``chat``).  Without either, ``-hf coder``.
"""

from __future__ import annotations

import argparse
import os
import sys

from deepinfer.cli import bench, chat, serve

COMMANDS = {
    "cli": (chat, "chat with a model in the terminal, or answer one prompt (-p)", ("run", "chat")),
    "server": (serve, "serve a page and an API over a context of up to millions of tokens", ("serve",)),
    "bench": (bench, "measure prompt processing and generation speed", ()),
}


def version() -> str:
    import deepinfer

    return deepinfer.__version__


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("-m", "--model", default=None, help="path to a GGUF file")
    group.add_argument("-hf", "--hf-repo", dest="hf", default=None,
                       help="Hugging Face repo[:QUANT] to fetch the GGUF from, or a preset: coder, chat "
                            "(default: coder = DeepSeek-Coder-V2-Lite-Instruct Q4_K_M)")
    group.add_argument("--meta", default=None, help="folder with the model's config.json and tokenizer files")
    group.add_argument("--meta-hf", default=None, help="Hugging Face repo to fetch config and tokenizer from")
    group.add_argument("--device", default="cuda:0", help="CUDA device (default %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepinfer", description="DeepInfer-SM86: DeepSeek-V2-Lite GGUF models with long contexts on one GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"deepinfer-sm86 {version()}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (module, summary, aliases) in COMMANDS.items():
        command = sub.add_parser(name, aliases=list(aliases), help=summary, description=module.__doc__.splitlines()[0])
        add_model_arguments(command)
        module.add_arguments(command)
        command.set_defaults(handler=module.run)
    pull = sub.add_parser("pull", help="download a model and its config/tokenizer files")
    add_model_arguments(pull)
    pull.set_defaults(handler=_pull)
    info = sub.add_parser("info", help="show what this machine offers: Python, PyTorch, GPU, Triton, CuPy")
    info.set_defaults(handler=_info)
    return parser


def _pull(args) -> None:
    from deepinfer.cli.models import resolve

    model = resolve(args.model, args.hf, args.meta, args.meta_hf)
    print(f"model:  {model.gguf}\nconfig: {model.meta}\ntrained context: {model.trained:,} tokens")


def _info(_args) -> None:
    import importlib.util
    import platform

    from deepinfer.cli.models import cache_dir

    print(f"deepinfer-sm86 {version()}")
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda or 'none'})")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            print(f"cuda:{index} {properties.name}, sm_{properties.major}{properties.minor}, "
                  f"{total / 2**30:.1f} GiB ({free / 2**30:.1f} free)")
    else:
        print("cuda: no GPU visible - the runtime needs an NVIDIA GPU")
    for module, role in (("triton", "fused long-context locate"), ("cupy", "fused MoE kernels, graph decoding")):
        found = importlib.util.find_spec(module) is not None
        print(f"{module}: {'yes' if found else 'no'} ({role}{'' if found else ', slower fallbacks'})")
    print(f"model cache: {cache_dir()}")


def main(argv: list[str] | None = None) -> None:
    from deepinfer.cli.models import DownloadError

    # the model's own modeling code draws transformers warnings about APIs it never uses
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except DownloadError as error:
        raise SystemExit(
            f"deepinfer: {error}\n  try again later, set HF_ENDPOINT to another mirror, or download the files "
            "yourself and pass -m FILE (and --meta DIR)"
        ) from None


def _alias(command: str):
    def entry() -> None:
        main([command, *sys.argv[1:]])

    return entry


cli_main = _alias("cli")
server_main = _alias("server")
bench_main = _alias("bench")


if __name__ == "__main__":
    main()
