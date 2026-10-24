# qwen3-tts-vc-demo

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（オープンウェイト版）の動作検証用の CLI スクリプト集です。
ボイスクローンを中心に、次の3種類のモデルをローカル GPU で動かし、処理時間・RTF・VRAM を計測します。

| モデル | できること | スクリプト |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B / 0.6B-Base | 数秒の参照音声と書き起こしから声をクローンする | `spike_clone.py` |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 自然文で書いた声の説明から音声を作る | `spike_design.py` |
| Qwen3-TTS-12Hz-1.7B / 0.6B-CustomVoice | 9人のプリセット話者で読み上げる（1.7B はスタイル指示にも対応） | `spike_custom.py` |

計測結果、ハマった点、生成パラメータの一覧は **[README_spike.md](README_spike.md)** にまとめています。

## 検証結果の要点

RTX 4090 / WSL2 / SDPA での結果です。

- 5モデルすべてが動作した。日本語・英語ともに生成でき、日本語の参照音声から英語を話させることもできる。
- **RTF は約 1.35〜1.7 で、実時間より遅い。** 律速は自己回帰ループのオーバーヘッドなので、0.6B にしても速くならない。
- VRAM は 1.7B が1モデル約 4〜5 GiB、0.6B が約 2〜3 GiB（生成中のピーク）。1.7B の3モデルを同時にロードしても約 12.2 GiB に収まる。
- flash-attn は SDPA より約 30% 遅かったため、既定では使わない。

## 動作環境

- NVIDIA GPU（検証は RTX 4090 24 GB。1.7B 1モデルなら VRAM 6 GB 程度で動く見込み）
- Linux または WSL2
- [uv](https://docs.astral.sh/uv/)
- Python 3.12（uv が自動で用意する）

検証時のバージョンは torch 2.14.1+cu130、qwen-tts 0.1.1、transformers 4.57.3 です。

## セットアップ

```bash
git clone https://github.com/m0t0k3y-hub/qwen3-tts-vc-demo.git
cd qwen3-tts-vc-demo
uv sync
```

モデルの重みは、初回実行時に Hugging Face Hub から自動でダウンロードされます（1.7B は約 4.3 GB、0.6B は約 2.4 GB）。

## 使い方

結果は `outputs/` に WAV（24 kHz）で保存され、計測値は `outputs/metrics.jsonl` に追記されます。

### ボイスクローン

```bash
uv run spike_clone.py <参照音声> "<参照音声の書き起こし>" "<合成するテキスト>" \
  --language Japanese --size 1.7B
```

- 参照音声は 5〜10 秒程度で、書き起こしは正確なほど良い結果になります。
- `--size 0.6B` で軽量モデルを使います。
- `--x-vector-only` を付けると書き起こしなしでも動きますが、品質は下がります。

### VoiceDesign

```bash
uv run spike_design.py "落ち着いた低めの声の40代男性。ゆっくり丁寧に話す。" "本日はお越しいただきありがとうございます。" \
  --language Japanese
```

### CustomVoice

```bash
uv run spike_custom.py --list    # 話者と言語の一覧
uv run spike_custom.py --speaker Ono_Anna --instruct "楽しそうに" --language Japanese "こんにちは。"
```

### VRAM 計測（1.7B の3モデルを同時にロード）

```bash
uv run spike_vram.py
```

### 共通オプション

| オプション | 説明 |
|---|---|
| `--language` | `Japanese` / `English` / `Auto` など |
| `--seed` | 乱数シード |
| `--out` | 出力パス |
| `--attn` | `sdpa`（既定）/ `flash_attention_2` / `eager` |
| `--temperature`、`--top-k`、`--top-p`、`--repetition-penalty`、`--max-new-tokens`、`--no-sample` | 生成パラメータ |
| `--subtalker-temperature`、`--subtalker-top-k`、`--subtalker-top-p`、`--subtalker-no-sample` | sub-talker（2番目以降のコードブック）の生成パラメータ |

既定値と各パラメータの意味は [README_spike.md](README_spike.md#生成パラメータ一覧プレイグラウンド用) を参照してください。

### 検証ケースの一括実行

```bash
./run_cases.sh design     # VoiceDesign 日英 4パターン
./run_cases.sh custom     # CustomVoice 日英 各1話者
./run_cases.sh vram       # 3モデル同時ロード
./run_cases.sh clone      # クローン（下記の参照音声が必要）
uv run summarize_metrics.py   # 計測結果を Markdown の表で出力
```

`clone` を実行するには、自分の声を次の名前で `samples/` に置いてください。`samples/` は `.gitignore` で除外されているので、コミットされません。

```
samples/ja.wav   日本語の参照音声（5〜10 秒）
samples/ja.txt   その書き起こし
samples/en.wav   英語の参照音声（5〜10 秒）
samples/en.txt   その書き起こし
```

## 注意

- ボイスクローンは、本人の同意を得た声だけに使ってください。
- 生成音声の音量は、参照音声の録音レベルをそのまま引き継ぎます。小さい場合は参照音声を正規化してください。

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) です。Qwen3-TTS のモデルと `qwen-tts` パッケージは、それぞれのライセンス（Apache-2.0）に従います。
