"""The deepinfer command: subcommands, llama.cpp-style aliases and options."""

from __future__ import annotations

import sys

import pytest

import deepinfer
from deepinfer.cli import bench, chat, serve
from deepinfer.cli import main as cli


def test_subcommands_and_their_aliases_route_to_the_tools() -> None:
    parser = cli.build_parser()
    for argv, handler in (
        (["cli", "-p", "hi"], chat.run),
        (["run", "-p", "hi"], chat.run),
        (["server", "-hf", "chat"], serve.run),
        (["serve"], serve.run),
        (["bench", "-p", "512,4096"], bench.run),
    ):
        assert parser.parse_args(argv).handler is handler


def test_llama_cpp_style_options() -> None:
    args = cli.build_parser().parse_args(
        ["server", "-hf", "coder:Q8_0", "-c", "1048576", "--port", "9000", "--api-key", "k"]
    )
    assert (args.hf, args.ctx_size, args.port, args.api_key, args.host) == ("coder:Q8_0", 1048576, 9000, "k", "127.0.0.1")
    args = cli.build_parser().parse_args(["cli", "-m", "x.gguf", "-n", "64", "-sys", "be brief", "--meta", "m"])
    assert (args.model, args.n_predict, args.system, args.meta, args.ctx_size) == ("x.gguf", 64, "be brief", "m", 32768)
    args = cli.build_parser().parse_args(["bench"])
    assert (args.n_prompt, args.n_gen, args.repetitions, args.no_warmup, args.device) == ("512", 128, 3, False, "cuda:0")


def test_the_alias_commands_prepend_their_subcommand(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(cli, "main", seen.append)
    monkeypatch.setattr(sys, "argv", ["deepinfer-server", "-hf", "chat", "-c", "65536"])
    cli.server_main()
    monkeypatch.setattr(sys, "argv", ["deepinfer-cli", "-p", "hi"])
    cli.cli_main()
    assert seen == [["server", "-hf", "chat", "-c", "65536"], ["cli", "-p", "hi"]]


def test_version_and_info(capsys) -> None:
    with pytest.raises(SystemExit) as done:
        cli.main(["--version"])
    assert done.value.code == 0 and deepinfer.__version__ in capsys.readouterr().out
    cli.main(["info"])
    out = capsys.readouterr().out
    assert f"deepinfer-sm86 {deepinfer.__version__}" in out and "model cache:" in out


def test_download_trouble_ends_in_one_message(monkeypatch) -> None:
    from deepinfer.cli.models import DownloadError

    def fail(args) -> None:
        raise DownloadError("could not download org/x from https://hf-mirror.com: 429 Client Error")

    monkeypatch.setattr(chat, "run", fail)
    with pytest.raises(SystemExit, match="429 Client Error\n  try again later"):
        cli.main(["cli", "-p", "hi"])


def test_a_command_is_required(capsys) -> None:
    with pytest.raises(SystemExit) as done:
        cli.main([])
    assert done.value.code == 2
