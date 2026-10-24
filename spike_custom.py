"""Preset-speaker TTS with Qwen3-TTS CustomVoice (1.7B / 0.6B).

Examples:
  uv run spike_custom.py --list
  uv run spike_custom.py --speaker Ono_Anna --language Japanese --instruct "嬉しそうに" "こんにちは、今日はいい天気ですね。"
"""

import argparse

from spike_common import (MODEL_IDS, add_common_args, gen_kwargs, load_model, log_metrics,
                          save_wav, seed_everything, synthesize, warmup_text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="text to synthesize")
    p.add_argument("--list", action="store_true", help="print supported speakers / languages and exit")
    p.add_argument("--speaker", default="Ono_Anna")
    p.add_argument("--instruct", default=None, help="style instruction (1.7B only; ignored by 0.6B)")
    p.add_argument("--language", default="Auto", help="Japanese / English / Auto ...")
    p.add_argument("--size", default="1.7B", choices=["1.7B", "0.6B"])
    add_common_args(p)
    args = p.parse_args()
    if not args.list and not args.text:
        p.error("text is required unless --list is given")

    model, load_m = load_model(MODEL_IDS[("custom_voice", args.size)], args.attn)

    print("[speakers]", ", ".join(model.get_supported_speakers() or []))
    print("[languages]", ", ".join(model.get_supported_languages() or []))
    if args.list:
        return
    if args.instruct and args.size == "0.6B":
        print("[warn] 0.6B-CustomVoice does not support instruct; it will be ignored")

    kw = gen_kwargs(args)
    warmup = None if args.no_warmup else dict(
        text=warmup_text(args.language), language=args.language, speaker=args.speaker,
        instruct=args.instruct, **kw)

    seed_everything(args.seed)
    wav, sr, synth_m = synthesize(
        model.generate_custom_voice, warmup,
        text=args.text, language=args.language, speaker=args.speaker, instruct=args.instruct, **kw)

    stem = f"custom_{args.size}_{args.speaker.lower()}_{args.language.lower()}"
    out = save_wav(wav, sr, args.out, stem)
    log_metrics({"script": "custom", "size": args.size, "speaker": args.speaker, "language": args.language,
                 "instruct": args.instruct, "text": args.text, "gen_kwargs": kw, "seed": args.seed,
                 "out": str(out), **load_m, **synth_m})


if __name__ == "__main__":
    main()
