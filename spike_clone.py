"""Voice clone with Qwen3-TTS Base (1.7B / 0.6B).

Example:
  uv run spike_clone.py samples/ja.wav "参照音声の書き起こし" "合成したいテキスト" --language Japanese
"""

import argparse
from pathlib import Path

from spike_common import (MODEL_IDS, add_common_args, gen_kwargs, load_model, log_metrics,
                          save_wav, seed_everything, synthesize, warmup_text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref_audio", help="reference audio (local path / URL)")
    p.add_argument("ref_text", help="transcript of the reference audio (ignored with --x-vector-only)")
    p.add_argument("text", help="text to synthesize")
    p.add_argument("--language", default="Auto", help="target language, e.g. Japanese / English / Auto")
    p.add_argument("--size", default="1.7B", choices=["1.7B", "0.6B"])
    p.add_argument("--x-vector-only", action="store_true",
                   help="use speaker embedding only (no transcript needed, lower quality)")
    add_common_args(p)
    args = p.parse_args()

    model, load_m = load_model(MODEL_IDS[("base", args.size)], args.attn)

    # Build the reference prompt once; it is reused for warm-up and the measured run.
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=None if args.x_vector_only else args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )
    kw = gen_kwargs(args)
    warmup = None if args.no_warmup else dict(
        text=warmup_text(args.language), language=args.language,
        voice_clone_prompt=prompt, **kw)

    seed_everything(args.seed)
    wav, sr, synth_m = synthesize(
        model.generate_voice_clone, warmup,
        text=args.text, language=args.language, voice_clone_prompt=prompt, **kw)

    stem = f"clone_{args.size}_{Path(args.ref_audio).stem}_{args.language.lower()}"
    out = save_wav(wav, sr, args.out, stem)
    log_metrics({"script": "clone", "size": args.size, "ref_audio": args.ref_audio,
                 "language": args.language, "text": args.text, "x_vector_only": args.x_vector_only,
                 "gen_kwargs": kw, "seed": args.seed, "out": str(out), **load_m, **synth_m})


if __name__ == "__main__":
    main()
