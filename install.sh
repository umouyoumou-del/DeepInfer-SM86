#!/bin/sh
# DeepInfer-SM86 installer for Linux and WSL (an NVIDIA GPU with a CUDA 12 driver).
#
#   curl -fsSL https://raw.githubusercontent.com/umouyoumou-del/DeepInfer-SM86/main/install.sh | sh
#
# Puts the commands deepinfer, deepinfer-cli, deepinfer-server and deepinfer-bench in
# ~/.local/bin, backed by a private Python environment in ~/.deepinfer; nothing else on the
# system changes.  Run it again to update.  Optional settings:
#
#   DEEPINFER_HOME    where the environment goes                 (~/.deepinfer)
#   DEEPINFER_BIN     where the commands go                      (~/.local/bin)
#   DEEPINFER_REF     branch, tag or commit to install           (main)
#   DEEPINFER_SOURCE  install this instead: a checkout, an archive or a URL
#   PYTHON            the Python 3.10+ to build the environment with
#   TORCH_INDEX_URL   where PyTorch comes from (default: picked from the driver's CUDA version);
#                     a Python that already has a CUDA build of PyTorch lends it instead
#   DEEPINFER_NO_MODIFY_PATH=1   leave ~/.bashrc and ~/.zshrc alone
set -eu

REPO="umouyoumou-del/DeepInfer-SM86"
REF="${DEEPINFER_REF:-main}"
HOME_DIR="${DEEPINFER_HOME:-$HOME/.deepinfer}"
BIN_DIR="${DEEPINFER_BIN:-$HOME/.local/bin}"
VENV="$HOME_DIR/venv"
COMMANDS="deepinfer deepinfer-cli deepinfer-server deepinfer-bench"

say() { printf 'deepinfer: %s\n' "$*"; }
die() { printf 'deepinfer: error: %s\n' "$*" >&2; exit 1; }

find_python() {
    for candidate in ${PYTHON:-python3.12 python3.11 python3.10 python3 python}; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# whether $1 imports a CUDA build of PyTorch 2.5 or newer; with "base", also that $1 is not a venv
cuda_torch() {
    "$1" - "${2:-}" <<'EOF' >/dev/null 2>&1
import sys
import torch
version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
in_venv = sys.prefix != sys.base_prefix
sys.exit(0 if torch.version.cuda and version >= (2, 5) and not (sys.argv[1] == "base" and in_venv) else 1)
EOF
}

# the wheel index of the newest PyTorch CUDA build a driver for CUDA $1.$2 runs
torch_index() {
    if [ "$1" -lt 12 ]; then return 1
    elif [ "$1" -gt 12 ] || [ "$2" -ge 8 ]; then tag=cu128
    elif [ "$2" -ge 6 ]; then tag=cu126
    elif [ "$2" -ge 4 ]; then tag=cu124
    else tag=cu121
    fi
    echo "https://download.pytorch.org/whl/$tag"
}

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        die "need curl or wget to download $1"
    fi
}

mkdir -p "$HOME_DIR"
if [ ! -x "$VENV/bin/python" ]; then  # an update reuses the environment and needs no other Python
    PY=$(find_python) || die "no Python 3.10 or newer found: install one (e.g. apt install python3 python3-venv) or set PYTHON"
    if cuda_torch "$PY" base; then
        say "creating $VENV, sharing the PyTorch of $PY"
        "$PY" -m venv --system-site-packages "$VENV" || { rm -rf "$VENV"; die "could not create a virtual environment with $PY (Debian/Ubuntu: apt install python3-venv)"; }
    else
        say "creating $VENV with $PY"
        "$PY" -m venv "$VENV" || { rm -rf "$VENV"; die "could not create a virtual environment with $PY (Debian/Ubuntu: apt install python3-venv)"; }
    fi
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

if ! cuda_torch "$VPY"; then
    index="${TORCH_INDEX_URL:-}"
    if [ -z "$index" ]; then
        cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -n 1)
        if [ -z "$cuda" ]; then
            say "warning: no NVIDIA driver found (nvidia-smi); the commands install, but running a model needs an NVIDIA GPU"
        else
            index=$(torch_index $cuda) ||
                die "the NVIDIA driver supports CUDA $(echo "$cuda" | tr ' ' .); DeepInfer needs CUDA 12 (driver 525 or newer): update the driver or set TORCH_INDEX_URL"
        fi
    fi
    if [ -n "$index" ]; then
        say "installing PyTorch from $index (about 3 GB)"
        "$VPY" -m pip install "torch>=2.5" --index-url "$index"
    fi
fi

SOURCE="${DEEPINFER_SOURCE:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
if [ -z "$SOURCE" ]; then
    say "downloading $REPO@$REF"
    SOURCE="$TMP/DeepInfer-SM86.zip"
    download "https://github.com/$REPO/archive/$REF.zip" "$SOURCE"
fi
case "$SOURCE" in
    *://*) url="$SOURCE" ;;
    *) url=$("$VPY" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve().as_uri())' "$SOURCE") ;;
esac
say "installing deepinfer-sm86 and its dependencies"
"$VPY" -m pip install --no-cache-dir "deepinfer-sm86[cuda] @ $url"  # no cache: a rebuilt archive keeps its name

mkdir -p "$BIN_DIR"
for name in $COMMANDS; do
    ln -sf "$VENV/bin/$name" "$BIN_DIR/$name"
done
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        if [ "${DEEPINFER_NO_MODIFY_PATH:-}" != 1 ]; then
            line="export PATH=\"$BIN_DIR:\$PATH\"  # DeepInfer-SM86"
            added=""
            for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
                if [ -f "$rc" ]; then
                    grep -qsF "$line" "$rc" || printf '\n%s\n' "$line" >>"$rc"
                    added="$added $rc"
                fi
            done
            [ -n "$added" ] && say "added $BIN_DIR to PATH in$added"
        fi
        say "for this shell: export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

"$BIN_DIR/deepinfer" info || true
cat <<EOF

deepinfer-sm86 is installed: $COMMANDS ($BIN_DIR)
  deepinfer-cli -hf coder                      chat in the terminal (the first run downloads 10 GB)
  deepinfer-cli -m model.gguf -p "a question"  one answer from a local GGUF
  deepinfer-server -hf coder                   page and API at http://127.0.0.1:8080/
  deepinfer-bench -hf coder -p 512,4096        prompt and generation speed
EOF
