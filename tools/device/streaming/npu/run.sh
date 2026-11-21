#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
    printf '%s\n' 'Usage:' '  ./run.sh --preflight [--install-app]' '  ./run.sh --serial USB_SERIAL [--install-app]' '  SERIAL=USB_SERIAL ./run.sh [--install-app]' 'Environment: ADB=/path/to/adb overrides local platform-tools; --serial overrides SERIAL.' 'Set MURMUR_NPU_BUNDLE to the staged directory. --preflight does not need adb.'
}
die() { printf '%s\n' "$*" >&2; exit 2; }
serial=${SERIAL:-}
explicit=0
preflight=0
app=0
while (($#)); do
    case $1 in
        --serial) (($# >= 2 && !explicit)) || die 'Give --serial exactly once with its USB serial'; serial=$2; explicit=1; shift 2 ;;
        --preflight) preflight=1; shift ;;
        --install-app) app=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown argument: $1" ;;
    esac
done
if ((preflight)); then
    ((!explicit)) || die '--serial cannot be combined with --preflight'
else
    [[ $serial =~ ^[A-Za-z0-9][A-Za-z0-9_:.%-]*$ && $serial != emulator-* ]] || die 'An explicit physical device serial is required'
fi
bundle=${MURMUR_NPU_BUNDLE:?set MURMUR_NPU_BUNDLE to a local staged bundle}
bundle=$(realpath -e -- "$bundle")
if ((!preflight && app)) && [[ ${MURMUR_STREAM_SERIAL:-} == "$serial" ]]; then
    die "App install is USB-only"
fi
for tool in timeout python3; do command -v "$tool" >/dev/null || die "Missing local tool: $tool"; done
args=()
if ((app)); then args+=(--app); fi
timeout -k 2s 25s python3 -I "$root/check.py" --bundle "$bundle" preflight "${args[@]}"
if ((preflight)); then exit 0; fi
adb_bin=${ADB:-}
if [[ -z $adb_bin ]]; then
    sdk=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/.local/share/android-sdk}}
    if [[ -x $sdk/platform-tools/adb ]]; then adb_bin=$sdk/platform-tools/adb; else adb_bin=adb; fi
fi
adb_bin=$(command -v -- "$adb_bin") || die 'No local adb: set ADB to Android platform-tools adb'
[[ -x $adb_bin ]] || die 'adb is not an executable file'
[[ $adb_bin = /* ]] || adb_bin=$PWD/$adb_bin
mkdir -p "$bundle/runs"
out=$(mktemp -d "$bundle/runs/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf 'Report directory: %s\n' "$out"
rc=0
timeout -k 8s 540s bash "$root/device.sh" "$adb_bin" "$serial" "$out" "$app" "$bundle" || rc=$?
printf '%s\n' "$rc" > "$out/exit-code.txt"
if ((rc == 124 || rc == 137)); then printf '%s\n' 'Global device time limit reached; see partial logs.' >&2; fi
exit "$rc"
