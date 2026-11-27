import argparse
import json
import math
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from lact_llm.lact_model import LaCTSWIGLUConfig, LaCTForCausalLM


def get_wikitext2_tokens(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = tokenizer("\n\n".join(ds["train"]["text"]), return_tensors="pt")["input_ids"][0]
    val_ids = tokenizer("\n\n".join(ds["validation"]["text"]), return_tensors="pt")["input_ids"][0]
    return train_ids, val_ids


@torch.no_grad()
def eval_ppls(model, val_ids, lengths=(512, 1024, 2048, 4096, 8192), n_batches=8):
    model.eval()
    results = {}
    for seq_len in lengths:
        losses = []
        max_start = len(val_ids) - seq_len - 1
        step = max(1, max_start // n_batches)
        for i in range(n_batches):
            s = min(i * step, max_start)
            x = val_ids[s : s + seq_len].unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=x)
            losses.append(out.loss.float().item())
        results[str(seq_len)] = round(math.exp(sum(losses) / len(losses)), 2)
    model.train()
    return results


def train_one(mode, train_ids, val_ids, args):
    torch.manual_seed(args.seed)
    cfg = LaCTSWIGLUConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attn_heads=args.num_attn_heads,
        num_lact_heads=args.num_lact_heads,
        inter_multi=1,
        window_size=args.window_size,
        lact_chunk_size=args.chunk_size,
        use_muon=True,
        use_momentum=True,
        ttt_loss_type=mode,
        rls_cg_iters=args.cg_iters,
        rls_init_lambda=args.init_lambda,
        fp32_states=True,
        initializer_range=0.02,
    )
    model = LaCTForCausalLM(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return max(0.01, (step + 1) / args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.time()
    max_start = len(train_ids) - args.seq_len - 1

    print(f"\n================ {mode} ({args.steps} steps, T={args.seq_len}, C={args.chunk_size}) ================")
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, max_start, (args.batch_size,))
        batch = torch.stack([train_ids[s : s + args.seq_len] for s in starts]).cuda()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = (time.time() - t0) / 60.0
            print(f"[{mode}] step {step:4d}/{args.steps} | lr {sched.get_last_lr()[0]:.1e} | loss {loss.item():.3f} | {elapsed:.1f}m")

    ppls = eval_ppls(model, val_ids)
    elapsed = round((time.time() - t0) / 60.0, 2)
    print(f"  -> {mode} PPLs: {ppls} ({elapsed} min)")
    return {"ppls": ppls, "elapsed_min": elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["dot_product", "bayesian_leaky_rls", "bayesian_leaky_rls_diag"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_attn_heads", type=int, default=4)
    p.add_argument("--num_lact_heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cg_iters", type=int, default=5)
    p.add_argument("--init_lambda", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default="wikitext2_rls_results.json")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    args.vocab_size = tokenizer.vocab_size
    train_ids, val_ids = get_wikitext2_tokens(tokenizer)

    summary = {}
    for mode in args.modes:
        summary[mode] = train_one(mode, train_ids, val_ids, args)

    print("\n=== OFFICIAL REPO WIKITEXT-2 COMPARISON ===")
    print(json.dumps(summary, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
