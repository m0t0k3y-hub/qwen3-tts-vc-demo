#!/usr/bin/env bash
# Run the spike verification cases from docs/spike01_voice-clone.md.
#
# Expects reference audio with transcripts:
#   samples/ja.wav + samples/ja.txt   (Japanese, 5-10 s)
#   samples/en.wav + samples/en.txt   (English, 5-10 s)
#
# Usage: ./run_cases.sh [clone|design|custom|vram|all]   (default: all)
# Extra args after the case name are passed to every spike script (e.g. --attn sdpa).
# SUFFIX=_xxx is appended to output file names so runs with different settings do not overwrite each other.
set -euo pipefail
cd "$(dirname "$0")"

CASE="${1:-all}"; shift || true
EXTRA=("$@")
SEED=0
SUFFIX="${SUFFIX:-}"   # appended to output names, e.g. SUFFIX=_fa

JA_TEXT="本日はお忙しい中お集まりいただき、誠にありがとうございます。さっそくですが、新しいプロジェクトの進捗についてご報告いたします。"
EN_TEXT="Thanks for joining today. Let me walk you through the progress on our new project and what comes next."

run_clone() {
  for f in samples/ja.wav samples/ja.txt samples/en.wav samples/en.txt; do
    [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }
  done
  local ja_ref en_ref
  ja_ref="$(cat samples/ja.txt)"
  en_ref="$(cat samples/en.txt)"
  for size in 1.7B 0.6B; do
    uv run spike_clone.py samples/ja.wav "$ja_ref" "$JA_TEXT" --language Japanese --size $size --seed $SEED \
      --out outputs/clone_${size}_ja2ja${SUFFIX}.wav "${EXTRA[@]}"
    uv run spike_clone.py samples/en.wav "$en_ref" "$EN_TEXT" --language English --size $size --seed $SEED \
      --out outputs/clone_${size}_en2en${SUFFIX}.wav "${EXTRA[@]}"
    uv run spike_clone.py samples/ja.wav "$ja_ref" "$EN_TEXT" --language English --size $size --seed $SEED \
      --out outputs/clone_${size}_ja2en${SUFFIX}.wav "${EXTRA[@]}"
  done
}

run_design() {
  uv run spike_design.py "落ち着いた低めの声の40代男性。ゆっくり、丁寧で信頼感のある話し方。" "$JA_TEXT" \
    --language Japanese --seed $SEED --out outputs/design_ja_calm_male${SUFFIX}.wav "${EXTRA[@]}"
  uv run spike_design.py "明るく元気な20代女性。少し高めの声で、笑顔が伝わるような弾んだ話し方。" \
    "こんにちは！今日はとっても楽しみにしていました。一緒に頑張りましょうね！" \
    --language Japanese --seed $SEED --out outputs/design_ja_bright_female${SUFFIX}.wav "${EXTRA[@]}"
  uv run spike_design.py "A warm, deep male voice in his 50s, speaking slowly like a documentary narrator." "$EN_TEXT" \
    --language English --seed $SEED --out outputs/design_en_narrator${SUFFIX}.wav "${EXTRA[@]}"
  uv run spike_design.py "Young energetic female, fast-paced and excited, like a sports commentator." \
    "And she scores! What an incredible finish to an unbelievable match!" \
    --language English --seed $SEED --out outputs/design_en_excited${SUFFIX}.wav "${EXTRA[@]}"
}

run_custom() {
  uv run spike_custom.py --speaker Ono_Anna --language Japanese --instruct "楽しそうに、明るく" "$JA_TEXT" \
    --seed $SEED --out outputs/custom_1.7B_ono_anna_ja${SUFFIX}.wav "${EXTRA[@]}"
  uv run spike_custom.py --speaker Ryan --language English --instruct "Calm and confident, like a news anchor." \
    "$EN_TEXT" --seed $SEED --out outputs/custom_1.7B_ryan_en${SUFFIX}.wav "${EXTRA[@]}"
}

run_vram() {
  uv run spike_vram.py "${EXTRA[@]}"
}

case "$CASE" in
  clone) run_clone ;;
  design) run_design ;;
  custom) run_custom ;;
  vram) run_vram ;;
  all) run_design; run_custom; run_vram; run_clone ;;
  *) echo "unknown case: $CASE" >&2; exit 1 ;;
esac
