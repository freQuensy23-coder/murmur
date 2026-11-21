#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
adb_bin=$1
serial=$2
out=$3
install_app=$4
bundle=$5
[[ $serial =~ ^[A-Za-z0-9][A-Za-z0-9_:.%-]*$ && $serial != emulator-* ]]
[[ $out =~ /runs/[0-9]{8}T[0-9]{6}Z\.([A-Za-z0-9]{6})$ ]]
run_id=${out##*/}
remote=/data/local/tmp/murmur-stream-npu-$run_id
unset ADB_SERVER_SOCKET ANDROID_ADB_SERVER_ADDRESS ANDROID_ADB_SERVER_PORT ADB_SERVER_PORT ANDROID_SERIAL
export ADB_MDNS_AUTO_CONNECT=0
adb_call() {
    local limit=$1; shift
    timeout --foreground -k 2s "${limit}s" "$adb_bin" -s "$serial" "$@"
}
sh_device() {
    local limit=$1; shift
    adb_call "$limit" shell -n "$@"
}
fail() { printf '%s\n' "$*" | tee -a "$out/errors.txt" >&2; exit 1; }
cleanup() {
    local rc=$?
    trap - EXIT
    sh_device 5 "rm -rf '$remote'" > "$out/cleanup.log" 2>&1 || printf 'Cleanup unavailable: %s\n' "$remote" >> "$out/cleanup.log"
    exit "$rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
printf '%s\n' "$serial" > "$out/serial.txt"
state=$(adb_call 10 get-state 2> "$out/transport.log") || fail 'ADB could not reach the named device serial'
[[ ${state//$'\r'/} == device ]] || fail "Unexpected ADB state: $state"
if [[ ${MURMUR_STREAM_SERIAL:-} == "$serial" && -n ${MURMUR_STREAM_RESERVATION:-} ]]; then
    printf '%s\n' 'Explicit Device Streaming transport' >> "$out/transport.log"
else
    transport=$(adb_call 8 get-devpath 2>> "$out/transport.log") || fail 'Could not identify the named transport'
    [[ $transport == usb:* ]] || fail "Not a locally attached USB transport: $transport"
fi
sh_device 12 'for p in ro.product.manufacturer ro.product.model ro.product.device ro.soc.model ro.soc.manufacturer ro.board.platform ro.product.cpu.abilist64 ro.kernel.qemu ro.boot.qemu ro.build.version.release ro.build.version.sdk ro.build.fingerprint ro.vendor.build.fingerprint; do printf "%s=%s\n" "$p" "$(getprop "$p")"; done; printf "page_size=%s\n" "$(getconf PAGE_SIZE 2>/dev/null)"; df -k /data; ls -l /vendor/lib64/libcdsprpc.so /vendor/lib64/rfs/dsp/snap /dev/fastrpc-cdsp 2>/dev/null; command -v timeout' | tr -d '\r' > "$out/device.txt" || fail 'Device facts or Android timeout utility unavailable'
grep -Eiq '^ro\.soc\.model=SM8750(-AC)?$' "$out/device.txt" || fail 'Rejected: expected SM8750(/-AC), not a different Snapdragon'
grep -Eq '^ro\.product\.cpu\.abilist64=([^,]+,)*arm64-v8a(,.*)?$' "$out/device.txt" || fail 'Rejected: arm64-v8a unavailable'
if grep -Eq '^ro\.(kernel|boot)\.qemu=1$' "$out/device.txt"; then fail 'Rejected: emulator'; fi
grep -qx '/system/bin/timeout' "$out/device.txt" || fail 'Android /system/bin/timeout unavailable'
for file in sha256sum timeout; do sh_device 7 "test -x /system/bin/$file" || fail "Android /system/bin/$file missing"; done
sh_device 8 '/system/bin/timeout -k 2 3 /system/bin/true' || fail 'Android timeout did not accept the required syntax'
sh_device 8 "umask 077; mkdir '$remote'" || fail 'Could not create isolated scratch directory'
trap cleanup EXIT
printf 'Remote scratch: %s\n' "$remote" > "$out/scratch.txt"
adb_call 210 push -z any "$bundle/payload/." "$remote/" > "$out/push.log" 2>&1 || fail 'Core artifact push failed'
awk '/  payload\// {sub(/  payload\//,"  "); print}' "$bundle/SHA256SUMS" > "$out/device.SHA256SUMS"
adb_call 12 push "$out/device.SHA256SUMS" "$remote/SHA256SUMS" >> "$out/push.log" 2>&1 || fail 'Remote checksum manifest push failed'
sh_device 20 "cd '$remote' && /system/bin/sha256sum -c SHA256SUMS && chmod 700 bin/probe" > "$out/device-artifacts.log" 2>&1 || fail 'Remote checksum or mode verification failed'

run_case() {
    local case_id=$1 context=$2 shape=$3 work mode status limit
    work=$out/$case_id
    mkdir -p "$work"
    local prefix="cd '$remote/$case_id' && export LD_LIBRARY_PATH='$remote/lib' ADSP_LIBRARY_PATH='$remote/lib;/vendor/lib64/rfs/dsp/snap;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp' &&"
    local plugin="$remote/lib/libonnxruntime_providers_qnn.so" backend="$remote/lib/libQnnHtp.so"
    local probe="$remote/bin/probe" args
    for mode in cpu qnn proof broken; do
        status=0
        limit=42
        case $mode in
            cpu) args="cpu cpu.onnx input.f32 $shape 4 7 cpu.f32";;
            qnn) args="qnn $context input.f32 $shape 4 7 qnn.f32 $plugin $backend";;
            proof) args="proof $context input.f32 $shape 4 1 proof.f32 $plugin $backend";;
            broken) args="qnn $context input.f32 $shape 4 1 broken.f32 $plugin $remote/no-such-LibQnnHtp.so"; limit=18;;
        esac
        sh_device "$((limit + 6))" "$prefix exec /system/bin/timeout -k 2 $limit $probe $args" > "$work/$mode.log" 2>&1 || status=$?
        printf '%s\n' "$status" > "$work/$mode.rc"
        if [[ $mode == broken ]]; then
            if ! sh_device 8 "test ! -e '$remote/$case_id/broken.f32'" >> "$work/broken.log" 2>&1; then
                printf '%s\n' 'Negative output absent check failed' >> "$work/broken.log"
                touch "$work/broken.f32"
            fi
        elif ((status == 0)); then
            adb_call 12 pull "$remote/$case_id/$mode.f32" "$work/$mode.f32" >> "$work/pull.log" 2>&1 || return 1
            if [[ $mode == proof ]]; then
                adb_call 12 pull "$remote/$case_id/proof.json" "$work/proof.json" >> "$work/pull.log" 2>&1 || return 1
                adb_call 12 pull "$remote/$case_id/htp.csv" "$work/htp.csv" >> "$work/pull.log" 2>&1 || return 1
            fi
        else
            printf '%s %s failed (rc=%s); see %s\n' "$case_id" "$mode" "$status" "$work/$mode.log" >&2
        fi
    done
}

app_smoke() {
    local air wifi installed base existing resumed
    air=$(sh_device 7 'settings get global airplane_mode_on') || return 1
    wifi=$(sh_device 7 'settings get global wifi_on') || return 1
    if [[ ${air//$'\r'/} != 1 || ${wifi//$'\r'/} != 0 ]]; then
        printf '%s\n' 'App phase refused: enable airplane mode and turn Wi-Fi off first; runner will not change settings.'
        return 1
    fi
    existing=$(sh_device 8 'pm list packages -u --user 0 app.murmur') || return 1
    if grep -Eq '^package:app\.murmur[[:space:]]*$' <<< "$existing"; then
        printf '%s\n' 'App phase refused: app.murmur is already installed for user 0; no replacement or uninstall performed.'
        return 1
    fi
    installed=$(adb_call 70 install --user 0 --no-streaming "$bundle/app/murmur-npu-debug.apk") || { printf '%s\n' "$installed"; return 1; }
    printf '%s\n' "$installed"
    grep -qx 'Success' <<< "${installed//$'\r'/}" || return 1
    base=$(sh_device 8 'pm path --user 0 app.murmur') || return 1
    base=${base//$'\r'/}
    [[ $base =~ ^package:(/data/app/[A-Za-z0-9_=/~.-]+/base\.apk)$ ]] || return 1
    sh_device 15 "sha256sum '${BASH_REMATCH[1]}'" > "$out/installed-apk.sha256" || return 1
    [[ $(awk '{print $1}' "$out/installed-apk.sha256") == "$(awk '$2=="app/murmur-npu-debug.apk" {print $1}' "$bundle/SHA256SUMS")" ]] || return 1
    sh_device 28 '/system/bin/timeout -k 2 22 am start -W --user 0 -n app.murmur/app.murmur.MainActivity' | tr -d '\r' | tee "$out/app-launch.txt" || return 1
    grep -qx 'Status: ok' "$out/app-launch.txt" || return 1
    sh_device 8 'pidof app.murmur' > "$out/app-pid.txt" || return 1
    resumed=$(sh_device 12 "dumpsys activity activities | grep -iE 'mResumedActivity|topResumedActivity' | grep -F 'app.murmur/'") || return 1
    [[ -n $resumed ]] || return 1
    printf '%s\n' "$resumed" > "$out/app-resumed.txt"
}

rc=0
run_case det det_h960_w416.onnx 1,3,960,416 || rc=1
run_case rec rec_w320.onnx 1,3,48,320 || rc=1
python3 -I "$root/check.py" --bundle "$bundle" evaluate "$out" | tee "$out/result.json" || rc=1
if ((install_app)); then
    if app_smoke > "$out/app.log" 2>&1; then
        printf 'PASS installed and resumed app.murmur\n' | tee "$out/app-status.txt"
    else
        printf 'FAIL app phase; see app.log\n' | tee "$out/app-status.txt" >&2
        rc=1
    fi
else
    printf 'NOT REQUESTED\n' > "$out/app-status.txt"
fi
exit "$rc"
