#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd "$(dirname "$0")/../../../.." && pwd)
assets=$(realpath -m "${1:?usage: build-test.sh ASSETS_DIRECTORY}")
mkdir -p "$assets/stream"
cd "$root"
python3 tools/e2e/fleurs_clip.py "$assets/stream" ru
./gradlew "-Pmurmur.streamAssets=$assets" :app:assembleDebugAndroidTest :testapps:target:assembleDebug
