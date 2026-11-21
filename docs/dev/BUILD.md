# Build Murmur

Requirements: JDK 21, Android SDK 37.2 (`compileSdk`), build-tools 37.0.0.

```bash
export ANDROID_HOME=$HOME/.local/share/android-sdk
./gradlew assembleDebug testDebugUnitTest
```

The speech and correction engines are native libraries built from source and not committed, so
an APK made without them installs but cannot dictate:

```bash
JOBS=$(nproc) native/sherpa/build-android.sh all
NO_HEAVY=1 JOBS=$(nproc) native/llama/build_android.sh cpu-arm64 cpu-x86_64
./gradlew assembleDebug          # picks up .cache/native/jniLibs
tools/e2e/check_apk_engines.sh app/build/outputs/apk/debug/app-debug.apk
```

CI does the same before it builds (`native-libs` job, cached) and fails the build when an APK
is missing an engine library. The standard CI builds carry the CPU engines only. The complete device
build adds the Qualcomm NPU paths and is published as separate `*-npu-*` release assets, built
locally because the Qualcomm toolchains need a licence-accepting account (ADR-048):

```bash
tools/device/build_full_apk.sh        # NPU + QNN libraries, verified, into dist/
```

Model weights are never committed.
