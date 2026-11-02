#!/bin/bash
# Fetch llama.cpp b12024 (ddf48f8) and build the CUDA tools for SM89.
set -euo pipefail
COMMIT=ddf48f85c933c1f438212a02eb21e7b001d2d37d
WORK=/root/autodl-tmp/llamacpp-vs-deepinfer
SRC=$WORK/llama.cpp-$COMMIT
mkdir -p "$WORK"; cd "$WORK"
if [ ! -d "$SRC" ]; then
  (. /etc/network_turbo >/dev/null; curl -sSL --retry 3 -o llama.cpp-b12024.tar.gz \
     "https://github.com/ggml-org/llama.cpp/archive/$COMMIT.tar.gz")
  sha256sum llama.cpp-b12024.tar.gz
  tar xzf llama.cpp-b12024.tar.gz
fi
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH
date; echo "== configure"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF 2>&1 | tail -40
echo "== targets"; cmake --build build --target help | grep -oE "llama-(bench|cli|simple|completion)\b" | sort -u
date; echo "== build"
T=$(cmake --build build --target help | grep -oE "llama-(bench|cli|simple|completion)\b" | sort -u | tr '\n' ' ')
cmake --build build -j 16 --target $T 2>&1 | tail -30
date; ls -la build/bin/
echo BUILD_DONE
