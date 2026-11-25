#!/usr/bin/env bash
# Fetch open-licence (OFL / Apache-2.0) Android, Cyrillic, Latin, and Hebrew fonts
# into .cache/ml/ocr_train/fonts/ for synthetic OCR crop rendering.
# Idempotent: skips existing valid TTF/OTF files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FONTS="${MURMUR_OCR_TRAIN_FONTS:-$ROOT/.cache/ml/ocr_train/fonts}"
GF="https://github.com/google/fonts/raw/main"

mkdir -p "$FONTS"

fetch() {
  local dest="$FONTS/$1"
  local url="$2"
  if [ -s "$dest" ]; then
    echo "ok   $1"
    return 0
  fi
  if curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"; then
    # Verify TrueType/OpenType magic bytes (00010000 or OTTO or ttcf)
    local magic
    magic="$(od -An -tx1 -N4 "$dest.part" | tr -d ' \n')"
    if [[ "$magic" == "00010000" || "$magic" == "4f54544f" || "$magic" == "74746366" ]]; then
      mv "$dest.part" "$dest"
      echo "got  $1"
      return 0
    fi
    rm -f "$dest.part"
    echo "warn invalid font magic ($magic) for $1" >&2
  else
    rm -f "$dest.part"
    echo "warn failed to fetch $1 ($url)" >&2
  fi
  return 1
}

# Hebrew + Latin + Cyrillic variable & static OFL fonts
fetch "NotoSansHebrew[wdth,wght].ttf" "$GF/ofl/notosanshebrew/NotoSansHebrew%5Bwdth,wght%5D.ttf" || true
fetch "Rubik[wght].ttf"                "$GF/ofl/rubik/Rubik%5Bwght%5D.ttf" || true
fetch "Heebo[wght].ttf"                "$GF/ofl/heebo/Heebo%5Bwght%5D.ttf" || true
fetch "Assistant[wght].ttf"            "$GF/ofl/assistant/Assistant%5Bwght%5D.ttf" || true
fetch "OpenSans[wdth,wght].ttf"        "$GF/ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf" || true
fetch "IBMPlexSansHebrew-Regular.ttf"  "$GF/ofl/ibmplexsanshebrew/IBMPlexSansHebrew-Regular.ttf" || true
fetch "IBMPlexSansHebrew-Medium.ttf"   "$GF/ofl/ibmplexsanshebrew/IBMPlexSansHebrew-Medium.ttf" || true
fetch "IBMPlexSansHebrew-Bold.ttf"     "$GF/ofl/ibmplexsanshebrew/IBMPlexSansHebrew-Bold.ttf" || true
fetch "FrankRuhlLibre[wght].ttf"       "$GF/ofl/frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf" || true
fetch "VarelaRound-Regular.ttf"        "$GF/ofl/varelaround/VarelaRound-Regular.ttf" || true

# Android core Latin + Cyrillic + Mono fonts (matching ml/ocr/fetch_assets.sh)
fetch "Roboto[wdth,wght].ttf"          "$GF/ofl/roboto/Roboto%5Bwdth,wght%5D.ttf" || true
fetch "NotoSans[wdth,wght].ttf"        "$GF/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf" || true
fetch "RobotoMono[wght].ttf"           "$GF/ofl/robotomono/RobotoMono%5Bwght%5D.ttf" || true
fetch "JetBrainsMono[wght].ttf"        "$GF/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf" || true

# Also symlink any fonts already cached by ml/ocr/fetch_assets.sh
SCREENS_FONTS="${MURMUR_SCREENS_FONTS:-$ROOT/.cache/ml/screens/fonts}"
if [ -d "$SCREENS_FONTS" ]; then
  for f in "$SCREENS_FONTS"/*.ttf; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    [[ "$base" == *Emoji* ]] && continue
    [ -e "$FONTS/$base" ] || ln -s "$f" "$FONTS/$base"
  done
fi

count="$(find "$FONTS" -maxdepth 1 \( -name '*.ttf' -o -name '*.otf' -o -name '*.ttc' \) | wc -l | tr -d ' ')"
echo "Fonts ready in $FONTS ($count font files)"
