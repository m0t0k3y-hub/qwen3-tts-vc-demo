# Spike 01: Qwen3-TTS 動作検証（ボイスクローン中心）

指示書: [docs/spike01_voice-clone.md](docs/spike01_voice-clone.md)

## 結論

- Base（クローン）1.7B / 0.6B、VoiceDesign 1.7B、CustomVoice 1.7B / 0.6B の 5モデルはすべて RTX 4090 + WSL2 で動作した。
- **RTF はモデルの種類・サイズによらず約 1.35〜1.5（SDPA）で、実時間より遅い。** 律速は GPU の計算量ではなく、
  自己回帰ループのオーバーヘッド（Python 処理とカーネル起動）。0.6B にしても速くならない。
- VRAM は 1.7B 1モデルあたり約 4 GiB（生成中のピーク reserved は約 4.1〜4.9 GiB）。0.6B は約 2〜3 GiB。
  **1.7B の3モデルを同時にロードしても 11.8 GiB、生成中のピークで 12.2 GiB**。24 GB に十分収まる。
- flash-attn はソースビルドで導入できたが、**SDPA より約 30% 遅く、VRAM も減らなかった**。既定は SDPA にしている。
- 本人の声によるクローン（日→日、英→英、日→英 × 1.7B / 0.6B）の6ケースを生成した。生成音声は参照音声の録音レベル
  （ピーク -15〜-17 dBFS）をそのまま引き継ぐため、音量が小さめになる。

## 環境

| 項目 | 値 |
|---|---|
| OS | Ubuntu 24.04.2 on WSL2（kernel 6.18.33.2-microsoft-standard-WSL2） |
| CPU / RAM | Intel Core i9-14900K（32 threads）/ 62 GB |
| GPU | RTX 4090 24 GB（sm_89） |
| ドライバ | Windows 側 610.57.01（`/usr/lib/wsl/lib` のライブラリ名から判断）。CUDA Driver API 13.5 |
| CUDA toolkit（WSL 内） | 13.2（`/usr/local/cuda-13.2`。flash-attn のビルドにのみ使用） |
| Python | 3.12.10（uv 0.7.8、`.python-version` で固定） |
| 主要パッケージ | qwen-tts 0.1.1、torch 2.14.1+cu130、torchaudio 2.11.0+cu130、transformers 4.57.3（qwen-tts が固定）、cuDNN 9.24 |

**nvidia-smi について**: `nvidia-smi` は `--version` を含むすべてのオプションでセグフォルトした（`LD_LIBRARY_PATH` を外しても同じ）。
WSL 側のドライバ配布物（`/usr/lib/wsl/lib/nvidia-smi`）の問題とみられる。ただし CUDA 自体は正常に動いており、
バージョン情報は PyTorch と `cuDriverGetVersion` で取得した。VRAM は `torch.cuda` の統計で計測している（後述）。

## セットアップ

```bash
uv sync                      # qwen-tts, torch, torchaudio（Python 3.12）
# 任意: flash-attn（遅くなるので非推奨。ビルド済み wheel を wheels/ に保存してある）
uv pip install wheels/flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl
```

モデルは初回実行時に HF Hub から `~/.cache/huggingface` へ自動でダウンロードされる（1.7B 約 4.3 GB、0.6B 約 2.4 GB）。

## スクリプト

| ファイル | 内容 |
|---|---|
| `spike_common.py` | モデルのロード、計測（ロード時間・合成時間・音声長・RTF・ピーク VRAM）、WAV 保存、`outputs/metrics.jsonl` への記録 |
| `spike_clone.py` | ボイスクローン。`ref_audio ref_text text --language --size {1.7B,0.6B} [--x-vector-only]` |
| `spike_design.py` | VoiceDesign。`instruct text --language` |
| `spike_custom.py` | CustomVoice。`--list` で話者と言語の一覧を表示。`--speaker --instruct --language --size text` |
| `spike_vram.py` | 1.7B の3モデルを順にロードし、各段階と同時ロード時の VRAM を計測。続けて各モデルで1回ずつ生成する |
| `run_cases.sh` | 指示書の検証ケースを一括で実行する（`clone|design|custom|vram|all`、追加の引数は各スクリプトへ渡す。`SUFFIX=_x` で出力名にサフィックスを付ける） |
| `summarize_metrics.py` | `outputs/metrics.jsonl` を Markdown の表として出力する |

全スクリプト共通のオプション: `--attn {sdpa,flash_attention_2,eager,auto}`、`--seed`、`--out`、`--no-warmup`、
生成パラメータ（`--temperature` など。後述）。

```bash
uv run spike_clone.py samples/ja.wav "$(cat samples/ja.txt)" "合成したいテキスト" --language Japanese --size 0.6B
uv run spike_design.py "落ち着いた低めの声の40代男性。" "こんにちは。" --language Japanese
uv run spike_custom.py --speaker Ono_Anna --instruct "楽しそうに" --language Japanese "こんにちは。"
uv run spike_vram.py
```

### 計測方法

- **ロード時間**: `Qwen3TTSModel.from_pretrained` の所要時間（キャッシュ済みの状態。初回はダウンロード込みで 200〜360 秒）。
- **合成時間**: `generate_*` 1回の所要時間（前後で `cuda.synchronize`）。クローンの参照プロンプト作成は計測に含めない。
  計測の前に、短い文で1回ウォームアップ生成を行う（`--no-warmup` で無効化）。
- **RTF** = 合成時間 / 生成音声の長さ（1未満なら実時間より速い）。
- **VRAM**: `torch.cuda.max_memory_allocated / max_memory_reserved`（PyTorch のアロケータ）と、
  `torch.cuda.mem_get_info` によるデバイス全体の使用量（CUDA コンテキストや Windows 側の使用分も含む）。
  何もロードしていない状態でも、デバイス全体では 1.53 GiB が使用済みだった（Windows デスクトップなど）。

## 結果

出力 WAV はすべて 24 kHz モノラル。サンプリングは既定値（`do_sample=True`）、seed 0。

### クローン（Base）0.6B vs 1.7B — 本人の声

参照音声: `samples/ja.wav`（9.56 秒）、`samples/en.wav`（10.88 秒）。どちらも 44.1 kHz ステレオで、
内部でモノラル化と 24 kHz へのリサンプルが行われる。合成テキストは `run_cases.sh` の `JA_TEXT` / `EN_TEXT`。

| 参照 → 合成 | size | 合成 s | 音声 s | RTF | ピーク alloc GiB | ピーク reserved GiB | 出力 |
|---|---|---:|---:|---:|---:|---:|---|
| 日 → 日 | 1.7B | 14.23 | 8.48 | 1.68 | 4.47 | 4.86 | `outputs/clone_1.7B_ja2ja.wav` |
| 日 → 日 | 0.6B | 12.13 | 8.88 | 1.37 | 2.59 | 3.06 | `outputs/clone_0.6B_ja2ja.wav` |
| 英 → 英 | 1.7B | 6.89 | 5.12 | 1.35 | 4.41 | 4.79 | `outputs/clone_1.7B_en2en.wav` |
| 英 → 英 | 0.6B | 10.47 | 6.56 | 1.60 | 2.57 | 2.97 | `outputs/clone_0.6B_en2en.wav` |
| 日 → 英 | 1.7B | 8.19 | 5.84 | 1.40 | 4.40 | 4.74 | `outputs/clone_1.7B_ja2en.wav` |
| 日 → 英 | 0.6B | 9.04 | 6.40 | 1.41 | 2.52 | 2.93 | `outputs/clone_0.6B_ja2en.wav` |

- attn は SDPA、seed 0。ロード時間（キャッシュ済み）は 1.7B / 0.6B ともに約 5〜6 秒。
- RTF は 1.35〜1.68 でばらつくが、サイズによる差ははっきりしない。0.6B の利点は VRAM が約 1.8 GiB 少ないこと。
- 参照音声が 10 秒前後あるため、ICL のプロンプトが長くなり、ピーク VRAM は公式サンプル（8 秒）の場合より少し大きい。
- 出力のピーク振幅は約 0.09〜0.14 と小さい。参照音声の録音レベル（ja: ピーク 0.18、en: 0.15）を引き継いでいるため。
  プレイグラウンドでは、参照音声を事前に正規化するか、出力を正規化する処理を入れるとよい（未検証）。

### クローン（Base）0.6B vs 1.7B — 公式サンプル音声（英語 8.08 秒）を参照に使用

| 参照 → 合成 | size | attn | 合成 s | 音声 s | RTF | ピーク alloc GiB | ピーク reserved GiB |
|---|---|---|---:|---:|---:|---:|---:|
| 英 → 英 | 1.7B | sdpa | 8.16 | 5.52 | 1.48 | 4.34 | 4.62 |
| 英 → 英 | 0.6B | sdpa | 7.21 | 5.20 | 1.39 | 2.44 | 2.72 |
| 英 → 日 | 1.7B | sdpa | 13.36 | 9.60 | 1.39 | 4.46 | 4.88 |
| 英 → 日 | 0.6B | sdpa | 13.25 | 9.68 | 1.37 | 2.57 | 2.97 |
| 英 → 英 | 1.7B | flash_attention_2 | 9.25 | 5.20 | 1.78 | 4.33 | 4.62 |
| 英 → 英 | 0.6B | flash_attention_2 | 9.82 | 4.64 | 2.12 | 2.43 | 2.65 |

- ロード時間（キャッシュ済み）は 1.7B / 0.6B ともに約 5〜6 秒。
- 重み常駐分は 1.7B が 3.91 GiB、0.6B が 2.02 GiB。
- 0.6B は VRAM がほぼ半分になるが、速度はほとんど変わらない。音質の比較は試聴して判断してほしい
  （`outputs/smoke_clone_*`）。

### VoiceDesign 1.7B

| ケース | attn | 合成 s | 音声 s | RTF | ピーク reserved GiB |
|---|---|---:|---:|---:|---:|
| 日: 落ち着いた40代男性 | sdpa | 17.43 | 12.80 | 1.36 | 4.49 |
| 日: 明るい20代女性 | sdpa | 14.60 | 8.40 | 1.74 | 4.24 |
| 英: ドキュメンタリーのナレーター | sdpa | 9.08 | 6.80 | 1.34 | 4.18 |
| 英: 興奮したスポーツ実況 | sdpa | 7.64 | 5.68 | 1.35 | 4.13 |

### CustomVoice

| ケース | size | attn | 合成 s | 音声 s | RTF | ピーク reserved GiB |
|---|---|---|---:|---:|---:|---:|
| 日: Ono_Anna +「楽しそうに、明るく」 | 1.7B | sdpa | 15.31 | 10.00 | 1.53 | 4.37 |
| 英: Ryan + "Calm and confident, like a news anchor." | 1.7B | sdpa | 7.89 | 5.84 | 1.35 | 4.13 |
| 日: Ono_Anna（instruct なし、上と同じ文） | 0.6B | sdpa | 17.81 | 12.88 | 1.38 | 2.58 |

- 話者（1.7B / 0.6B 共通、大文字小文字は区別しない）: `aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian`
- 言語: `auto, chinese, english, french, german, italian, japanese, korean, portuguese, russian, spanish`
- 0.6B-CustomVoice は instruct に対応していない（パッケージ側で黙って `None` に置き換えられる。スクリプトでは警告を出す）。

### 1.7B の3モデル同時ロード時の VRAM（SDPA。flash_attention_2 でも同じ値）

| 段階 | allocated GiB | reserved GiB | デバイス全体 GiB |
|---|---:|---:|---:|
| ベースライン（何もロードしていない） | 0.00 | 0.00 | 1.53 |
| + Base 1.7B | 3.91 | 4.04 | 5.64 |
| + VoiceDesign 1.7B | 7.81 | 7.93 | 9.53 |
| **+ CustomVoice 1.7B（3モデル常駐）** | **11.70** | **11.82** | **13.42** |
| 3モデル常駐 + 生成中のピーク | 12.09 | 12.21 | 13.83 |

→ 3モデル常駐のプレイグラウンドでも、24 GB の約 58% で収まる。0.6B を混ぜれば、さらに 2 GiB 程度ずつ減らせる。

### flash-attn と SDPA の比較

同じ seed・同じ文で交互に2回ずつ計測（CustomVoice 1.7B、Ryan、英文）した結果、SDPA は RTF 1.42 / 1.52、
flash_attention_2 は 1.93 / 1.95 だった。他のケースでも flash-attn は一貫して約 30% 遅く、VRAM の差は誤差の範囲だった。
1ステップ1トークンの自己回帰生成では attention の計算量自体が小さいうえ、HF の flash-attn 経路では
毎ステップ unpad/pad の前後処理が加わるためと考えられる。**既定は SDPA とした。**

## RTF が 1 を超える理由（プレイグラウンド設計への影響）

0.6B-CustomVoice の英文 1回（5.12 秒の音声）をプロファイルした結果:

- `speech_tokenizer.decode`（コーデックのデコード）は **0.09 秒** で、ほぼ無視できる。
- 残りはすべて talker の `generate`（8.47 秒）。12 Hz なので 1秒あたり 12 ステップで、各ステップで
  sub-talker（code predictor）が HF の `generate` を呼んで残り 15 コードブックを生成する。
  この例では talker 65 ステップに対し、sub-talker の forward が 960 回呼ばれていた。
- そのため演算量の小さい 0.6B でも速くならない。Python 処理とカーネル起動が支配的。

高速化の候補（今回は未検証）: 公式 README にある vLLM 経由での推論、CUDA Graph や `torch.compile`（static cache）の適用。
プレイグラウンドで対話的に使うなら、ストリーミング出力を検討するか、待ち時間を前提に UI を設計する必要がある。

## ハマった点

1. **nvidia-smi のセグフォルト**: 上記のとおり。CUDA の利用には影響しないため、情報は PyTorch から取得した。
2. **`uv init --bare` だと Python バージョンが固定されない**: `--python 3.12` を付けても `.python-version` が作られず、
   venv が 3.13 で作成された。`uv python pin 3.12` で固定し、venv を作り直した（公式の推奨は 3.12）。
3. **torch と torchaudio のバージョンが揃わない**: `uv add qwen-tts torch torchaudio` の結果は torch 2.14.1 + torchaudio 2.11.0 だった
   （torchaudio は torch と同じペースでリリースされなくなっている）。ただし qwen-tts で torchaudio を使うのは 25Hz トークナイザ
   （`speech_vq.py`）だけで、今回の 12Hz モデルの経路では使われないため問題なかった。import も通る。
4. **import 時の警告（無害）**: `SoX could not be found!` と `flash-attn is not installed` が表示される。どちらも 25Hz トークナイザ側の
   import によるもので、12Hz モデルには影響しない。sox 本体はインストールしていない。
5. **flash-attn のビルド**:
   - torch 2.14 / CUDA 13 用のビルド済み wheel は公式リリースにない（最新の 2.8.3.post2 でも cu12 + torch ≤2.8 まで）ため、ソースからビルドした。
   - そのままビルドすると `namespace "std" has no member "strong_ordering"` など大量のエラーが出る。torch 2.14 のヘッダは C++20 を要求するが、
     flash-attn の `setup.py` は `-std=c++17` を指定しているため。sdist を展開し、`setup.py` の `-std=c++17` を `-std=c++20` に置換して解決した。
   - ビルドのコマンド: `CUDA_HOME=/usr/local/cuda-13.2 MAX_JOBS=12 FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation <展開したディレクトリ>`
     （事前に `uv pip install setuptools wheel ninja packaging psutil` が必要。sm_80 だけに絞って約 15 分。sm_80 のバイナリは sm_89 でも動く）。
   - ビルドしたものの、前述のとおり遅くなったため使っていない。
6. **HF のダウンロードが遅い**: 1モデルあたり数分かかった（初回ロードの 200〜360 秒は大半がダウンロード）。
7. **README の記述と実装の差**: README には「HF Transformers の `generate` の kwargs（`max_new_tokens`、`top_p` など）を何でも渡せる」とあるが、
   実装（`modeling_qwen3_tts.py` の `generate`）では talker に渡す kwargs が固定の dict で組み立てられており、
   **下表以外の kwargs（例: `min_p`、`no_repeat_ngram_size`）は受け取っても無視される**。

## 生成パラメータ一覧（プレイグラウンド用）

ソース（`qwen_tts/inference/qwen3_tts_model.py`、`core/models/modeling_qwen3_tts.py`）と各モデルの `generation_config.json` で確認した。

### サンプリング（全モデル共通、`generate_*` の kwargs）

既定値は 5モデルとも同じ `generation_config.json` の値（未指定時に使われる）。

| パラメータ | 既定値 | 対象 | 説明 |
|---|---|---|---|
| `do_sample` | True | talker（第1コードブック） | False でグリーディ |
| `temperature` | 0.9 | talker | |
| `top_k` | 50 | talker | |
| `top_p` | 1.0 | talker | |
| `repetition_penalty` | 1.05 | talker | |
| `subtalker_dosample` | True | sub-talker（残り 15 コードブック） | |
| `subtalker_temperature` | 0.9 | sub-talker | |
| `subtalker_top_k` | 50 | sub-talker | |
| `subtalker_top_p` | 1.0 | sub-talker | |
| `max_new_tokens` | 8192 | talker | 12 Hz なので 8192 ≒ 約 11 分。config がない場合は 2048 |
| `eos_token_id` | config の codec EOS | talker | 通常は変更不要 |

内部で固定されている値: `min_new_tokens=2`、`suppress_tokens`（コーデック語彙の末尾 1024 のうち EOS 以外）。

スクリプトでの指定例: `--temperature 0.7 --top-k 30 --repetition-penalty 1.1 --subtalker-temperature 0.7 --max-new-tokens 2048 --no-sample`

**seed**: 同じ seed・同じ attn 実装なら同じ長さの音声が再現された（`--seed`。`torch.manual_seed` などを設定する）。

### モデル別の入力

| モデル | メソッド | 引数 |
|---|---|---|
| Base（1.7B / 0.6B） | `generate_voice_clone` | `text`、`language`、`ref_audio`（パス / URL / base64 / `(ndarray, sr)`）、`ref_text`、`x_vector_only_mode=False`、`voice_clone_prompt`、`non_streaming_mode=False` |
| Base | `create_voice_clone_prompt` | `ref_audio`、`ref_text`、`x_vector_only_mode`。参照の特徴量を一度だけ計算して使い回せる |
| VoiceDesign 1.7B | `generate_voice_design` | `text`、`instruct`（声の説明文。空文字も可）、`language`、`non_streaming_mode=True` |
| CustomVoice（1.7B / 0.6B） | `generate_custom_voice` | `text`、`speaker`、`language`、`instruct`（1.7B のみ。0.6B では無視される）、`non_streaming_mode=True` |
| CustomVoice | `get_supported_speakers()` / `get_supported_languages()` | 一覧（小文字） |

- **言語**: `"Auto"`（既定）または英語名（`Japanese`、`English` など。大文字小文字は区別しない）。公式は「分かっているなら明示する」ことを推奨。
- **参照音声の書き起こし（クローン）**: 既定の ICL モードでは `ref_text` が必須で、参照音声のコード列と書き起こしの両方を条件に生成する。
  `x_vector_only_mode=True` にすると話者埋め込みだけを使う（書き起こし不要。ただし品質は落ちる、と公式に記載）。
  参照音声は内部で 24 kHz にリサンプルされる。長さの制約は明記されていない（「3秒でクローン」とある）。
- **`non_streaming_mode`**: 公式の説明では「False の場合は、テキストのストリーミング入力を*模擬*するだけで、本当のストリーミング生成ではない」。
  既定値がクローン（False）と Design / Custom（True）で異なる。
- **instruct の書き方**: 自然文で指定する（中国語・英語の例が公式にある。日本語でも受け付けた）。
  例: `"Speak in an incredulous tone, but with a hint of panic"`、`"Male, 17 years old, tenor range, ..."`。
- **バッチ**: `text`、`language`、`speaker`、`instruct` はリストで渡すとバッチで推論する。
- 公式に紹介されている使い方: VoiceDesign で参照音声を作り、それを Base の `create_voice_clone_prompt` に渡すと、
  デザインした声を固定の話者として使い回せる。

## 未完了 / 次のステップ

- [x] 本人の声でのクローン（`./run_cases.sh clone`）。参照音声を差し替えた場合も同じコマンドで再実行し、
  `uv run summarize_metrics.py` で表を更新できる。
- [ ] 生成した WAV の試聴による品質評価（特に 1.7B と 0.6B の声の似かた、日→英のアクセント）。
  自動チェックでは、全ファイルで無音・NaN がなく、長さも妥当なことだけ確認した。
- [ ] 参照音声と出力の音量正規化の要否を確認する。
- [ ] 高速化（vLLM、torch.compile）の検証は別のスパイクで行う。
