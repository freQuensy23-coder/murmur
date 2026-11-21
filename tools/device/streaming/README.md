# Device streaming smoke tests

These tools operate on an explicitly selected, already attached device. They do not rent, discover or cancel devices. Use only a disposable test device, with the Android CLI managing the reservation separately. Verify the model and actual new ADB transport before running; cancel the exact reservation and check its terminal state when finished. Do not publish report directories, serials, payloads, or fixtures.

## App: WAV to editor

1. Build a debug app APK. Set `work="$PWD/.cache/stream-app"` and run `bash tools/device/streaming/app/build-test.sh "$work/assets"` to generate the ignored FLEURS fixture and compile the test and target APKs. The fixture fetch uses the pinned public FLEURS reference; no audio is checked in.
2. Run `python3 tools/device/streaming/app/stage.py "$work/bundle" --app-apk APP.apk --test-apk app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk --target-apk testapps/target/build/outputs/apk/debug/target-debug.apk --tier insertion`. `ocr` also needs the four catalog ASR/VAD/OCR models already present in `.cache/models` (or `--models`); `full` adds the cleanup model. Use `launch` for install/launch only.
3. Run `python3 tools/device/streaming/app/run.py "$work/bundle" --mode insertion --preflight`. This uses no ADB. For an attached test device replace `--preflight` with `--serial SERIAL --expect-model MODEL_PREFIX --disposable`; `--plan` only inspects the attached device, `--emulator` explicitly selects a local x86_64 rehearsal. Use `ADB` to override platform-tools. Keep generated output in the staging directory.

The test separates Practice from insertion into an external editor, requires COMMIT_TEXT, checks OCR readiness in the real tiers, and logs stop-to-insert timing. Start/stop in the external editor is instrumentation-injected; WAV playback never qualifies the microphone.

## QNN: same-device CPU versus strict HTP

Set `MURMUR_NPU_BUNDLE` to a private local directory outside the tracked tree. Supply ORT 1.32.0 arm64 and its compatible QNN runtime in `payload/lib/`; build `payload/bin/probe` with `npu/build-probe.sh`. Use the CPU ONNX and SM8750 contexts from the catalog: `payload/{det,rec}/cpu.onnx`, `payload/det/det_h960_w416{.onnx,_qnn.bin}` and `payload/rec/rec_w320{.onnx,_qnn.bin}`. Supply matching float32 little-endian NCHW `payload/{det,rec}/input.f32` (1×3×960×416 / 1×3×48×320), and host CPU outputs `reference/{det,rec}.f32` (1×1×960×416 / 1×40×852) produced by the same probe with a host ORT build. For optional USB-only install/launch supply `app/murmur-npu-debug.apk` and disable radios before starting; do not use it with a streaming transport.

In the bundle directory generate `SHA256SUMS` with `find payload reference -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS`; append `sha256sum app/murmur-npu-debug.apk >> SHA256SUMS` only when using that phase. Run `bash tools/device/streaming/npu/run.sh --preflight` before any device work. Then use `--serial SERIAL`; for a separately reserved stream explicitly set `MURMUR_STREAM_SERIAL` to the same attached serial and `MURMUR_STREAM_RESERVATION` to the live handle. Do not run this against an emulator. The harness requires QNN-only kernel events, positive accelerator execute timing and a failed missing-backend control, not just a successful APK launch.
