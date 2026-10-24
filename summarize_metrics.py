"""Print outputs/metrics.jsonl as Markdown tables (latest record per output file).

Example:
  uv run summarize_metrics.py
"""

import json

from spike_common import METRICS_PATH


def main():
    records = [json.loads(line) for line in METRICS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    synth = {r["out"]: r for r in records if r["script"] != "vram"}  # later runs overwrite earlier ones
    vram = [r for r in records if r["script"] == "vram"]

    print("| script | size | case | attn | load s | synth s | audio s | RTF | peak alloc GiB | peak reserved GiB |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(synth.values(), key=lambda r: (r["script"], r["size"], r["out"])):
        v = r["synth_vram"]
        case = r["out"].rsplit("/", 1)[-1].removesuffix(".wav")
        print(f"| {r['script']} | {r['size']} | {case} | {r['attn']} | {r['load_sec']:.1f} | {r['synth_sec']:.2f} | "
              f"{r['audio_sec']:.2f} | {r['rtf']:.2f} | {v['peak_allocated_gib']:.2f} | {v['peak_reserved_gib']:.2f} |")

    for r in vram:
        print(f"\nVRAM run {r['time']} (attn={r['attn']}):\n")
        print("| step | allocated GiB | reserved GiB | device used GiB |")
        print("|---|---:|---:|---:|")
        b = r["baseline"]
        print(f"| baseline | {b['allocated_gib']:.2f} | {b['reserved_gib']:.2f} | {b['device_used_gib']:.2f} |")
        for s in r["steps"]:
            v = s["after_load"]
            print(f"| + {s['kind']} | {v['allocated_gib']:.2f} | {v['reserved_gib']:.2f} | {v['device_used_gib']:.2f} |")
        for k, g in r["generate"].items():
            v = g["synth_vram"]
            print(f"| generate {k} (peak) | {v['peak_allocated_gib']:.2f} | {v['peak_reserved_gib']:.2f} | "
                  f"{v['device_used_gib']:.2f} |")


if __name__ == "__main__":
    main()
