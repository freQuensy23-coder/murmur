<div align="center">

<img src="fastlane/metadata/android/en-US/images/icon.png" alt="Иконка Murmur" width="88" />

# Murmur

**100% офлайн-диктовка для Android — без замены вашей клавиатуры.**

[![Последний релиз](https://img.shields.io/github/v/release/freQuensy23-coder/murmur?style=flat-square&color=6366f1&label=Release)](https://github.com/freQuensy23-coder/murmur/releases/latest)
[![Android 14+](https://img.shields.io/badge/Android-14%2B-3ddc84?style=flat-square&logo=android&logoColor=white)](https://github.com/freQuensy23-coder/murmur/releases/latest)
[![Лицензия](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)
[![Obtainium](https://img.shields.io/badge/Obtainium-Автообновления-7c3aed?style=flat-square)](https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/freQuensy23-coder/murmur)

[Скачать APK](https://github.com/freQuensy23-coder/murmur/releases/latest) · [Obtainium](https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/freQuensy23-coder/murmur) · [English](README.md)

<br />

<img src="docs/readme/demo.gif" alt="Демо офлайн-диктовки Murmur в авиарежиме" width="760" />

</div>

---

Нажмите плавающую кнопку в любом приложении, продиктуйте текст и нажмите ещё раз — чистый, расставленный по знакам ответ сразу появится в поле ввода. Распознавание речи, локальный OCR экрана и дообученная LLM-автоправка работают прямо на телефоне без интернета.

## Почему Murmur

- **Ваша клавиатура остаётся на месте** — плавающий пузырь работает поверх Telegram, Google Messages, Slack, Gmail и любых других приложений без замены Gboard или Samsung Keyboard.
- **100% офлайн** — ASR, VAD, опциональный OCR и LLM-автоправка работают на устройстве. Работает в авиарежиме, без аккаунтов и телеметрии.
- **Умная автоправка** — дообученная локальная LLM убирает слова-паразиты, расставляет пунктуацию и аккуратно чистит формулировки.
- **Контекст экрана только в RAM** — опциональный локальный OCR читает имена и термины с экрана только в оперативной памяти (ничего не пишется на диск и никуда не отправляется).
- **Выбор моделей и бенчмарки** — меняйте модели ASR и автоправки прямо в приложении и сравнивайте скорость CPU и Qualcomm NPU на своём телефоне.

## Установка

Требуется **Android 14+**.

- **APK напрямую:** скачайте подписанный `release.apk` в [GitHub Releases](https://github.com/freQuensy23-coder/murmur/releases/latest).
- **Автообновления:** добавьте репозиторий в [Obtainium](https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/freQuensy23-coder/murmur).

**Первый запуск:**
1. Выдайте разрешения на микрофон, показ поверх окон и службу специальных возможностей.
2. Скачайте пакет **Speech + Murmur Cleanup** по Wi-Fi (или выберите **Speech only**).
3. Нажмите в любое поле ввода и тапните по плавающей кнопке `m`.

## Внутри приложения

<p align="center">
  <img src="docs/readme/gallery.webp" alt="Экраны Murmur: знакомство, практика и хаб моделей" width="760" />
</p>

- **История** — прослушивание сохранённых записей, сравнение сырого и очищенного текста, повторная обработка другой моделью.
- **Хаб моделей** — загрузка, переключение и замер скорости движков на вашем устройстве.
- **Настройки** — выбор объёма контекста с экрана, стиля автоправки и поведения кнопки.

## Локальный стек

| Этап | Модели / движок | Ускорение |
| --- | --- | --- |
| **Распознавание речи (ASR)** | Parakeet TDT 0.6B v3, GigaAM v3, Nemotron Streaming (ONNX / Sherpa) | CPU и Qualcomm NPU |
| **Детектор речи (VAD)** | Silero VAD v5 | CPU |
| **LLM-автоправка** | Дообученные Qwen3.5 / LFM (GGUF через `llama.cpp`) | CPU и Qualcomm NPU |
| **OCR экрана** *(опционально)* | PP-OCRv5 / PP-OCRv6 (только в RAM) | CPU и Qualcomm NPU |

<details>
<summary><strong>Замеры на устройстве (Galaxy S25 Ultra)</strong></summary>

<br />

- Полная задержка от нажатия «Стоп» до вставки текста с включённой автоправкой: **~815 мс** на Galaxy S25 Ultra.
- Отдельный замер OCR экрана (медиана 7 прогретых прогонов, Qualcomm NPU без отката на CPU):
  - Детекция: **2.79 мс** на NPU против **34.40 мс** на CPU
  - Распознавание: **3.15 мс** на NPU против **12.41 мс** на CPU

<p align="center">
  <img src="docs/readme/ocr-times.png" alt="Замеры OCR на Galaxy S25 Ultra: CPU против NPU" width="640" />
</p>

</details>

## Документация и сборка

- [Архитектура и пайплайн](docs/ARCHITECTURE.md)
- [Сборка из исходников](docs/dev/BUILD.md)
- [Каталог моделей](core/catalog/src/main/resources/catalog/catalog.json)

## Лицензия

[Apache-2.0](LICENSE) · [Сторонние компоненты](THIRD_PARTY_NOTICES.md). У скачиваемых моделей свои лицензии.
