"""Load the three 1.7B models (Base, VoiceDesign, CustomVoice) together and measure VRAM.

After all three are resident, one short generation per model is run to see the
peak with inference on top of the resident weights.

Example:
  uv run spike_vram.py
"""

import argparse

import torch

from spike_common import (MODEL_IDS, fmt_vram, load_model, log_metrics, resolve_attn, synthesize,
                          vram_snapshot)

TEXT = "こんにちは。これはVRAM計測用のテスト音声です。"
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    p.add_argument("--ref-audio", default=REF_AUDIO)
    p.add_argument("--ref-text", default=REF_TEXT)
    p.add_argument("--no-generate", action="store_true", help="only measure resident weights")
    args = p.parse_args()

    torch.cuda.init()
    baseline = vram_snapshot()
    print(f"[baseline] {fmt_vram(baseline)}")

    models, steps = {}, []
    for kind in ["base", "voice_design", "custom_voice"]:
        models[kind], m = load_model(MODEL_IDS[(kind, "1.7B")], args.attn)
        steps.append({"kind": kind, "load_sec": m["load_sec"], "after_load": vram_snapshot()})

    resident = vram_snapshot()
    print(f"\n[all 3 resident] {fmt_vram(resident)}")

    gen = {}
    if not args.no_generate:
        _, _, gen["base"] = synthesize(models["base"].generate_voice_clone, text=TEXT, language="Japanese",
                                       ref_audio=args.ref_audio, ref_text=args.ref_text)
        _, _, gen["voice_design"] = synthesize(models["voice_design"].generate_voice_design, text=TEXT,
                                               language="Japanese", instruct="明るく元気な若い女性の声。")
        _, _, gen["custom_voice"] = synthesize(models["custom_voice"].generate_custom_voice, text=TEXT,
                                               language="Japanese", speaker="Ono_Anna")
        peak = max(g["synth_vram"]["peak_reserved_gib"] for g in gen.values())
        print(f"\n[peak during generation, all 3 resident] reserved {peak:.2f} GiB")

    print("\n=== summary (GiB) ===")
    print(f"{'step':<24}{'allocated':>10}{'reserved':>10}{'device':>10}")
    print(f"{'baseline':<24}{baseline['allocated_gib']:>10.2f}{baseline['reserved_gib']:>10.2f}"
          f"{baseline['device_used_gib']:>10.2f}")
    for s in steps:
        v = s["after_load"]
        print(f"{'+ ' + s['kind']:<24}{v['allocated_gib']:>10.2f}{v['reserved_gib']:>10.2f}{v['device_used_gib']:>10.2f}")
    for k, g in gen.items():
        v = g["synth_vram"]
        print(f"{'gen ' + k + ' (peak)':<24}{v['peak_allocated_gib']:>10.2f}{v['peak_reserved_gib']:>10.2f}"
              f"{v['device_used_gib']:>10.2f}")

    log_metrics({"script": "vram", "attn": resolve_attn(args.attn), "baseline": baseline,
                 "steps": steps, "resident": resident, "generate": gen})


if __name__ == "__main__":
    main()
