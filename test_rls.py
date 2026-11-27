import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lact_llm", "lact_model"))
from ttt_operation import (
    block_causal_lact_swiglu,
    block_causal_lact_swiglu_rls,
)


def test_ttt_ops():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[1/2] Testing ttt_operation kernels on {device} ({dtype})...")

    B, H, L, D, DH, C = 2, 2, 128, 32, 32, 32
    b = B * H

    for mode in ["dot_product", "bayesian_leaky_rls", "bayesian_leaky_rls_diag"]:
        torch.manual_seed(42)
        w0 = (torch.randn(b, DH, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        w1 = (torch.randn(b, D, DH, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        w2 = (torch.randn(b, DH, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)

        q = F.normalize(torch.randn(b, L, D, device=device, dtype=dtype), dim=-1).requires_grad_(True)
        k = F.normalize(torch.randn(b, L, D, device=device, dtype=dtype), dim=-1).requires_grad_(True)
        v = torch.randn(b, L, D, device=device, dtype=dtype, requires_grad=True)

        lr0 = F.softplus(torch.randn(b, L, 1, device=device, dtype=torch.float32) - 4.0).requires_grad_(True)
        lr1 = F.softplus(torch.randn(b, L, 1, device=device, dtype=torch.float32) - 4.0).requires_grad_(True)
        lr2 = F.softplus(torch.randn(b, L, 1, device=device, dtype=torch.float32) - 4.0).requires_grad_(True)
        mom = torch.sigmoid(torch.randn(b, L, 1, device=device, dtype=torch.float32)).requires_grad_(True)

        if mode == "dot_product":
            out = block_causal_lact_swiglu(
                w0, w1, w2, q, k, v, lr0, lr1, lr2,
                chunk_size=C, use_muon=True, momentum=mom,
            )
            check_tensors = {"w0": w0, "w1": w1, "w2": w2, "q": q, "k": k, "v": v}
        else:
            gamma_raw = (torch.randn(b, L, 1, device=device, dtype=torch.float32) + 3.0).requires_grad_(True)
            lam_raw = torch.full((b, 1, 1), -2.25, device=device, dtype=torch.float32, requires_grad=True)
            gamma = torch.sigmoid(gamma_raw)
            rls_lam = F.softplus(lam_raw)

            out = block_causal_lact_swiglu_rls(
                w0, w1, w2, q, k, v, lr0, lr1, lr2,
                gamma=gamma,
                rls_lambda=rls_lam,
                chunk_size=C,
                use_muon=True,
                momentum=mom,
                diag_precision=(mode == "bayesian_leaky_rls_diag"),
                cg_iters=5,
            )
            check_tensors = {
                "w0": w0, "w1": w1, "w2": w2, "q": q, "k": k, "v": v,
                "gamma_raw": gamma_raw, "lam_raw": lam_raw,
            }

        assert out.shape == (b, L, D), f"{mode}: bad shape {out.shape}"
        assert torch.isfinite(out).all(), f"{mode}: NaN/Inf in forward output"

        loss = out.float().pow(2).mean()
        loss.backward()

        for name, t in check_tensors.items():
            assert t.grad is not None, f"{mode}: missing grad for {name}"
            assert torch.isfinite(t.grad).all(), f"{mode}: NaN/Inf in grad for {name}"

        print(f"  PASS {mode:24s} | out={tuple(out.shape)} | loss={loss.item():.4f}")


def test_full_model_cuda():
    if not torch.cuda.is_available():
        print("[2/2] Skipping full LaCTForCausalLM test (no CUDA; run on cluster).")
        return

    print("[2/2] Testing full LaCTForCausalLM on CUDA...")
    from lact_llm.lact_model import LaCTSWIGLUConfig, LaCTForCausalLM

    for mode in ["dot_product", "bayesian_leaky_rls", "bayesian_leaky_rls_diag"]:
        cfg = LaCTSWIGLUConfig(
            vocab_size=1024,
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            inter_multi=1,
            window_size=64,
            lact_chunk_size=32,
            use_muon=True,
            use_momentum=True,
            ttt_loss_type=mode,
            rls_cg_iters=5,
            rls_init_lambda=0.1,
            fp32_states=True,
        )
        model = LaCTForCausalLM(cfg).cuda().to(torch.bfloat16)

        # Verify post_init() did not clobber gamma_proj.bias
        if mode.startswith("bayesian_leaky_rls"):
            b_val = model.model.layers[0].attn.gamma_proj.bias.data.mean().item()
            assert abs(b_val - 3.0) < 1e-4, f"{mode}: gamma_proj.bias was {b_val}, expected 3.0"

        x = torch.randint(0, cfg.vocab_size, (2, 128), device="cuda")
        out = model(input_ids=x, labels=x)
        assert torch.isfinite(out.loss), f"{mode}: NaN/Inf loss"
        out.loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{mode}: NaN/Inf grad in {n}"

        print(f"  PASS full model {mode:24s} | loss={out.loss.item():.4f}")


if __name__ == "__main__":
    test_ttt_ops()
    test_full_model_cuda()
