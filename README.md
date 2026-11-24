# Murmur

[По-русски](README.ru.md) · [Latest Releases](https://github.com/freQuensy23-coder/murmur/releases) · [Auto-updates via Obtainium](https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/freQuensy23-coder/murmur)

![Murmur: on-device dictation for Android 14 and later](docs/readme/cover.png)

**Leave the cursor where you want the words.** Tap Murmur’s floating bubble, dictate a reply, then tap again. Your words appear in the same field. Keep your usual keyboard. Once the models are downloaded, dictation works on the phone without an internet connection.

![Three illustrated stages in a chat: a recording bubble, a transcribing indicator, then a message in the composer](docs/readme/bubble-sequence.webp)
*How the bubble works, illustrated with the project’s chat prototype.*

## Try it on your phone

Install the [debug-signed test build](https://github.com/freQuensy23-coder/murmur/releases/tag/test-56e9d4b) on **Android 14 or newer**. Follow the setup prompts for the microphone, floating bubble and accessibility service. Choose **Speech only**, download the model pack on Wi-Fi, then try a sentence in the practice field before switching to a messaging app.

By default Murmur inserts the speech recognizer’s transcript. You can enable local model cleanup to add punctuation or use words visible on screen. **Cleanup is experimental:** it can change or omit spoken words. Screen context and dictated audio are processed on the device.

## Inside the app

![Cropped Android emulator screenshots: Welcome explains the bubble, Practice shows an inserted test sentence, and Models lists speech engines](docs/readme/gallery.webp)
*Android emulator captures with sample data. Open a full screen: [Welcome](docs/readme/welcome.webp) · [Practice](docs/readme/practice.webp) · [Models](docs/readme/models.webp).*

**History** lets you replay a saved recording and retry a dictation. **Models** lets you download engines and compare their speed on your own phone. **Settings** is where you choose what screen context cleanup can use.

## What has actually run on a phone?

On a **Galaxy S25 Ultra**, Murmur transcribed a prerecorded Russian sentence with no word errors, read nine words from the screen, and inserted the result into another app’s test field with Samsung Keyboard. With optional cleanup on, the field updated **815 ms after stop** in that run. The test used software start/stop and prerecorded audio; it does not establish microphone quality, bubble usability or Fold7 behavior.

![Standalone screen recognition on Galaxy S25 Ultra. Detection: CPU 34.40 ms, NPU 2.79 ms. Recognition: CPU 12.41 ms, NPU 3.15 ms.](docs/readme/ocr-times.png)
*Separate OCR benchmark, not whole-dictation latency: median of seven warm runs per engine on the same input. The Qualcomm accelerator ran without CPU fallback.*

[How the pipeline works](docs/ARCHITECTURE.md) · [Build from source](docs/dev/BUILD.md) · [Model catalog](core/catalog/src/main/resources/catalog/catalog.json) · [Physical-device checklist](docs/DEVICE_CHECKLIST.md)

[Apache-2.0](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md). Downloadable models retain their own licenses.
