"""Voice design with Qwen3-TTS 1.7B-VoiceDesign: natural-language voice description + text -> WAV.

Example:
  uv run spike_design.py "落ち着いた低めの声の30代男性。ゆっくり丁寧に話す。" "本日はお越しいただきありがとうございます。" --language Japanese
"""

import argparse

from spike_common import (MODEL_IDS, add_common_args, gen_kwargs, load_model, log_metrics,
                          save_wav, seed_everything, synthesize, warmup_text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instruct", help="natural-language description of the voice / speaking style")
    p.add_argument("text", help="text to synthesize")
    p.add_argument("--language", default="Auto", help="Japanese / English / Auto ...")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    add_common_args(p)
    args = p.parse_args()

    model, load_m = load_model(MODEL_IDS[("voice_design", "1.7B")], args.attn)

    kw = gen_kwargs(args)
    warmup = None if args.no_warmup else dict(
        text=warmup_text(args.language), language=args.language, instruct=args.instruct, **kw)

    seed_everything(args.seed)
    wav, sr, synth_m = synthesize(
        model.generate_voice_design, warmup,
        text=args.text, language=args.language, instruct=args.instruct, **kw)

    stem = f"design_{args.language.lower()}" + (f"_{args.tag}" if args.tag else "")
    out = save_wav(wav, sr, args.out, stem)
    log_metrics({"script": "design", "size": "1.7B", "language": args.language, "instruct": args.instruct,
                 "text": args.text, "gen_kwargs": kw, "seed": args.seed, "out": str(out), **load_m, **synth_m})


if __name__ == "__main__":
    main()
