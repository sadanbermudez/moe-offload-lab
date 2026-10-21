"""How many bytes must one decode step touch, and what speed does each bottleneck allow?

Reads the model's config.json and the shard sizes, splits the weights into routed experts (FP4, streamed
from disk when not resident) and everything else ("dense": attention, shared experts, norms, embeddings,
kept in RAM), then prints a ceiling in tokens/s for each resource:

    disk  : routed bytes that miss the RAM cache / measured disk bandwidth
    memory: all bytes touched per token / memory bandwidth
    a measured decode rate can then be placed against these ceilings.

Usage: python bytes_per_token.py <model_dir> --disk-gbps 4.85 --mem-gbps 273 --hit 0.08 [--measured 0.48]
"""
import argparse
import json
from pathlib import Path

FP4_BYTES = 0.5           # E2M1, two weights per byte
FP4_SCALE_BYTES = 1 / 32  # one UE8M0 scale byte per 32 weights (MX-style block scaling)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--disk-gbps", type=float, required=True, help="measured, e.g. by iobench")
    ap.add_argument("--mem-gbps", type=float, required=True, help="DRAM bandwidth (theoretical or measured)")
    ap.add_argument("--hit", type=float, default=0.0, help="share of routed expert reads served from RAM")
    ap.add_argument("--measured", type=float, default=None, help="observed decode tok/s, to place on the chart")
    a = ap.parse_args()

    c = json.loads((a.model_dir / "config.json").read_text())
    layers, h, inter = c["num_hidden_layers"], c["hidden_size"], c["moe_intermediate_size"]
    n_exp, k = c["n_routed_experts"], c["num_experts_per_tok"]
    assert c.get("expert_dtype") == "fp4", "this estimate assumes FP4 experts"

    params_per_expert = 3 * h * inter                          # gate, up, down projections
    expert_bytes = params_per_expert * (FP4_BYTES + FP4_SCALE_BYTES)
    routed_total = layers * n_exp * expert_bytes
    disk_total = sum(p.stat().st_size for p in a.model_dir.glob("*.safetensors"))
    dense_total = disk_total - routed_total                    # everything that is not a routed expert

    routed_per_tok = layers * k * expert_bytes
    from_disk_per_tok = routed_per_tok * (1 - a.hit)
    touched_per_tok = routed_per_tok + dense_total             # upper bound: all dense weights read each step

    GB = 1e9
    print(f"model            {c['model_type']}, {layers} layers, {n_exp} experts/layer, top-{k}")
    print(f"one expert       {params_per_expert / 1e6:.1f} M params, {expert_bytes / 1e6:.1f} MB at FP4")
    print(f"on disk          {disk_total / GB:.1f} GB = {routed_total / GB:.1f} GB routed experts + {dense_total / GB:.1f} GB dense")
    print(f"active / token   {routed_per_tok / GB:.2f} GB routed experts ({layers} x {k} x {expert_bytes / 1e6:.1f} MB)")
    print(f"from disk/token  {from_disk_per_tok / GB:.2f} GB at {a.hit:.0%} RAM hit rate")
    print()
    disk_ceiling = a.disk_gbps * GB / from_disk_per_tok if from_disk_per_tok else float("inf")
    mem_ceiling = a.mem_gbps * GB / touched_per_tok
    print(f"ceiling: disk    {disk_ceiling:6.2f} tok/s  ({a.disk_gbps} GB/s)")
    print(f"ceiling: memory  {mem_ceiling:6.2f} tok/s  ({a.mem_gbps} GB/s, all dense + active experts)")
    if a.measured:
        lowest = min(disk_ceiling, mem_ceiling)
        print(f"measured         {a.measured:6.2f} tok/s  = {a.measured / lowest:.0%} of the lowest ceiling; "
              f"implied disk read {a.measured * from_disk_per_tok / GB:.2f} GB/s")


if __name__ == "__main__":
    main()
