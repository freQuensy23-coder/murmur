#!/usr/bin/env bash
set -euo pipefail
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"
: "${ORT_HEADERS:?set ORT_HEADERS to the ONNX Runtime headers directory}"
: "${ORT_LIBS:?set ORT_LIBS to the arm64 ONNX Runtime library directory}"
out=${1:?usage: build-probe.sh OUTPUT}
source_dir=$(cd "$(dirname "$0")" && pwd)
compiler="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/${NDK_HOST_TAG:-linux-x86_64}/bin/aarch64-linux-android34-clang++"
"$compiler" -std=c++17 -O2 -I "$ORT_HEADERS" "$source_dir/probe.cc" -L "$ORT_LIBS" -lonnxruntime -o "$out"
