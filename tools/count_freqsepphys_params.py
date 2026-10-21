"""Count FreqSepPhys parameters (mock mamba CUDA deps)."""
import sys
import os
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAMBA_ROOT = os.path.join(ROOT, "tools", "mamba")
sys.path.insert(0, ROOT)
sys.path.insert(0, MAMBA_ROOT)

# --- mock optional compiled extensions ---
def _mock(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

cc = _mock("causal_conv1d")
cc.causal_conv1d_fn = None
cc.causal_conv1d_update = None

ssi = _mock("mamba_ssm.ops.selective_scan_interface")
ssi.selective_scan_fn = None
ssi.mamba_inner_fn = None
ssi.bimamba_inner_fn = None
ssi.mamba_inner_fn_no_out_proj = None

# Pre-register mamba_ssm submodules needed by mamba_simple
import importlib.util  # noqa: E402

def _load(rel_path, name):
    path = os.path.join(MAMBA_ROOT, rel_path.replace(".", os.sep) + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load("mamba_ssm.modules.mamba_simple", "mamba_ssm.modules.mamba_simple")
pkg = types.ModuleType("mamba_ssm")
pkg.Mamba = sys.modules["mamba_ssm.modules.mamba_simple"].Mamba
sys.modules["mamba_ssm"] = pkg

import torch  # noqa: E402
from neural_methods.model.FreqSepPhys import FreqSepPhys  # noqa: E402


def main():
    model = FreqSepPhys(theta=0.5, frames=128)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())

    print(f"Parameters (trainable weights): {total:,} ({total/1e6:.3f} M)")
    print(f"Buffers (fixed Gaussian, excluded from Params): {buffers:,}")

    print("\n--- Top-level ---")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:20s} {n:>12,}  ({n/1e6:.3f} M)")

    # 6 FreqSepBlocks total
    for i, blk_name in enumerate(["Block1", "Block2", "Block3", "Block4", "Block5", "Block6"], 1):
        blk = getattr(model, blk_name)
        n = sum(p.numel() for p in blk.parameters())
        print(f"  {blk_name} total: {n:,}")


if __name__ == "__main__":
    main()
