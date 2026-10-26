# qwen3-tts-vc-demo

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（オープンウェイト版）をローカル GPU で試すためのリポジトリです。
次の2つがあります。

- **[プレイグラウンド（Web UI）](#プレイグラウンドweb-ui)**: 3種類のモデルの機能をブラウザから一通り試せる Gradio アプリ。社内 LAN から複数人で使える。
- **[検証用 CLI スクリプト](#cli-スクリプト)**: 各モデルを1回ずつ動かし、処理時間・RTF・VRAM を計測する。

対象のモデルは次の3種類です。

| モデル | できること | スクリプト |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B / 0.6B-Base | 数秒の参照音声と書き起こしから声をクローンする | `spike_clone.py` |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 自然文で書いた声の説明から音声を作る | `spike_design.py` |
| Qwen3-TTS-12Hz-1.7B / 0.6B-CustomVoice | 9人のプリセット話者で読み上げる（1.7B はスタイル指示にも対応） | `spike_custom.py` |

プレイグラウンドでは、参照音声の自動書き起こしに Whisper large-v3（`openai/whisper-large-v3`）も使います。

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

モデルの重みは、初回に使うときに Hugging Face Hub から自動でダウンロードされます（1.7B は約 4.3 GB、0.6B は約 2.4 GB、Whisper large-v3 は約 3 GB）。

flash-attn は依存に含めていません（SDPA より遅かったため）。手で入れた場合、`uv sync` を実行すると外れます。入れ直す手順は [README_spike.md](README_spike.md#セットアップ) を参照してください。

## プレイグラウンド（Web UI）

3種類のモデルをブラウザから試せる Gradio アプリです。要件は [docs/playground.md](docs/playground.md) にまとめています。

```bash
uv run playground/app.py          # http://localhost:7860
```

- 生成は GPU 1枚で1件ずつ順に処理します。複数人が同時に使うと、後から投げた生成は待ち行列に入ります。
- モデルは使うときにロードし、VRAM の上限（既定 18 GiB）を超えそうになったら、最も長く使われていないモデルから解放します。
- 生成結果と条件は `playground_data/` に保存され、「履歴」タブで全員の結果を検索・再生・削除できます。計測値は `outputs/metrics.jsonl` にも追記されます。

| タブ | できること |
|---|---|
| ボイスクローン / VoiceDesign / CustomVoice | 各モデルで生成する。全生成パラメータを変えられ、例文・声の説明・スタイル指示の例を挿入できる。長文は文ごとに分割して生成でき、最大8行をまとめてバッチ推論することもできる |
| 話者ライブラリ | クローンした声や VoiceDesign で作った声を名前を付けて保存し、全員で使い回す |
| 比較 | 同じテキストを最大4条件で生成して聴き比べる。条件1との違いを表示する |
| 履歴 | 全員の生成結果を検索・再生・削除する。「この条件で開く」で、生成したときの条件をフォームに戻す。2〜4件を選んで比較タブに並べられる |
| システム | ロード中のモデルと VRAM、モデル別の RTF の集計と散布図、環境情報 |

ボイスクローンタブでは、参照音声の録音（ホスト PC から `http://localhost` で開いた場合のみ）、Whisper large-v3 による自動書き起こし（初回は約 3 GB のダウンロード）、前後の無音除去と音量の正規化ができます。

表示言語（日本語 / 英語）はブラウザの言語設定に従い、画面下の「設定」から切り替えられます。

CustomVoice タブの話者サンプル（9話者 × 日英）は、「システム」タブのボタンか次のコマンドで一度だけ生成してください（約2分）。

```bash
uv run playground/samples.py
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `7860` | 待ち受けアドレスとポート |
| `--preload` | `base:1.7B` | 起動時にロードするモデル（例: `base:1.7B custom_voice:0.6B`。`none` でロードしない） |
| `--vram-budget` | `18` | モデルを常駐させる VRAM の上限（GiB） |
| `--max-chars` | `500` | 1回に合成できる文字数の上限 |
| `--queue-size` | `20` | 待ち行列の上限 |
| `--target-lufs` | `-16` | 音量正規化の目標ラウドネス |
| `--data-dir` | `playground_data` | 履歴と音声の保存先 |

### 社内 LAN に公開する（WSL2）

WSL2 の中で起動したアプリに、LAN 上の他の PC からアクセスするには、次のどちらかを行います。

**方法1: mirrored ネットワーク（Windows 11 22H2 以降）**

Windows 側の `%UserProfile%\.wslconfig` に次を書き、PowerShell で `wsl --shutdown` してから WSL を起動し直します。

```ini
[wsl2]
networkingMode=mirrored
```

続けて、管理者の PowerShell で Hyper-V ファイアウォールと Windows ファイアウォールの受信を許可します。

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
New-NetFirewallRule -DisplayName "Qwen3-TTS Playground" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow
```

**方法2: ポートフォワード**

WSL の IP アドレス（WSL 内で `hostname -I`）に、Windows のポートを転送します。WSL の IP は再起動で変わるため、そのたびに設定し直してください。管理者の PowerShell で実行します。

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7860 connectaddress=<WSL の IP> connectport=7860
New-NetFirewallRule -DisplayName "Qwen3-TTS Playground" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow
```

他の PC からは `http://<Windows の IP アドレス>:7860` で開きます。

## CLI スクリプト

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

## 開発

```bash
uv run pytest            # テスト（GPU を使うものは GPU がない環境では自動でスキップ）
```

| パス | 内容 |
|---|---|
| `tts_core/` | モデル ID、ロード、計測、生成パラメータの定義。CLI とプレイグラウンドで共有する |
| `playground/engine.py` | 生成処理（入力チェック、参照音声の前処理、文分割・バッチ、履歴への記録） |
| `playground/model_manager.py` | モデルのオンデマンドロードと、VRAM 上限による解放（LRU） |
| `playground/history.py`、`speakers.py` | 履歴と話者ライブラリ（SQLite） |
| `playground/ui_*.py`、`app.py` | Gradio の画面。`app.py` が起動スクリプト |
| `playground/messages.py`、`i18n.py` | 画面に出すメッセージと日英の翻訳 |
| `spike_*.py`、`run_cases.sh` | 検証用 CLI |

## 注意

- ボイスクローンは、本人の同意を得た声だけに使ってください。
- 生成音声の音量は、参照音声の録音レベルをそのまま引き継ぎます。プレイグラウンドでは既定で出力を正規化しますが、CLI では正規化しないので、小さい場合は参照音声を正規化してください。

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) です。Qwen3-TTS のモデルと `qwen-tts` パッケージは、それぞれのライセンス（Apache-2.0）に従います。
