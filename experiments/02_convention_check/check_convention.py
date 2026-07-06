"""Step 2: spilled_energy() の入力規約検証
仮説: spilled_energy() は forward-pass 規約 (logits[k] = ids[0..k] を文脈とした出力,
ids[0]=BOS) を想定している。テストスクリプトが渡す generate() の scores
(scores[t] = ids[0..t-1] を文脈とした出力) では1ステップずれる。
"""
import sys, os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from spilled_energy.generation import generate_answer
from spilled_energy.energy import spilled_energy

MODEL = "meta-llama/Meta-Llama-3-8B"
QUESTION = "Who wrote the play Romeo and Juliet?"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    prompt = f"Q: {QUESTION}\nA:"
    out = generate_answer(prompt=prompt, model=model, tokenizer=tok,
                          max_new_tokens=100, do_sample=False, device="cuda")

    seqs = out["sequences"]              # [1, total_len] (プロンプト+BOS込み)
    scores = out["scores"]               # tuple 長さ n_steps, 各 [1, vocab]
    n_steps = len(scores)
    input_len = seqs.shape[1] - n_steps
    gen_ids = seqs[0, input_len:].tolist()
    full_ids = seqs[0].tolist()

    print(f"[1] input_len={input_len}, n_steps={n_steps}, total_len={len(full_ids)}")
    print(f"    先頭3トークン: {tok.convert_ids_to_tokens(full_ids[:3])}")

    # ---- forward pass(teacher forcing で全系列のlogitsを再計算)----
    with torch.no_grad():
        logits_fp = model(seqs).logits[0].float().cpu()   # [total_len, vocab]

    # ---- (A) 同一文脈のlogitsの一致確認: scores[t] ≟ logits_fp[input_len+t-1] ----
    diffs, agree = [], 0
    for t in range(n_steps):
        s = scores[t][0].float().cpu()
        f = logits_fp[input_len + t - 1]
        diffs.append((s - f).abs().max().item())
        agree += int(s.argmax().item() == f.argmax().item())
    print(f"[2] max|scores[t] - logits_fp[input_len+t-1]| = {max(diffs):.4f}")
    print(f"    argmax一致: {agree}/{n_steps}")

    # ---- (B-1) 現行方式: scores + 生成idのみ(テストスクリプトと同じ)----
    scores_list = torch.stack(scores, dim=1)[0].float().cpu().numpy().tolist()
    d_cur, Em_cur, E_cur = spilled_energy(logits=[scores_list], ids=[gen_ids], beta=1.0)

    # ---- (B-2) forward-pass 方式: 全系列logits + 全系列id(synth_maths.ipynb と同じ)----
    fp_list = logits_fp.numpy().tolist()   # 変換に十数秒かかります
    d_fp, Em_fp, E_fp = spilled_energy(logits=[fp_list], ids=[full_ids], beta=1.0)
    # 生成部分だけスライス(full位置 input_len+i ↔ gen位置 i)
    d_fp_gen  = d_fp[0][input_len:]
    E_fp_gen  = [float(x) for x in E_fp[0][input_len:]]
    Em_fp_gen = Em_fp[0][input_len:]
    E_cur_flat = [float(x) for x in E_cur[0]]

    print("\n[3] 生成トークン先頭8個 (i: token | E_cur | E_fwd | Δ_cur | Δ_fwd)")
    for i in range(min(8, n_steps)):
        t = tok.convert_ids_to_tokens([gen_ids[i]])[0]
        print(f"  {i:3d}: {t!r:>15} | {E_cur_flat[i]:8.3f} | {E_fp_gen[i]:8.3f} | "
              f"{float(d_cur[0][i]):8.3f} | {d_fp_gen[i]:8.3f}")

    # ---- (C) greedy恒等式: E_fwd[i] は -max(scores[i]) に一致するはず ----
    checks = [abs(E_fp_gen[i] + scores[i][0].float().max().item()) for i in range(n_steps)]
    print(f"\n[4] greedy検算 max|E_fwd[i] - (-max scores[i])| = {max(checks):.4f}")

    # ---- (D) 修正span [0,2) での exact answer メトリクス(forward版)----
    print("\n[5] exact answer 'William Shakespeare' (forward-pass版):")
    for i in range(2):
        t = tok.convert_ids_to_tokens([gen_ids[i]])[0]
        print(f"  {t!r}: E={E_fp_gen[i]:.4f}, E_margin={Em_fp_gen[i]:.4f}, delta={d_fp_gen[i]:.4f}")

    # ---- (E) 全体比較 ----
    cur = np.array([float(x) for x in d_cur[0]]); fp = np.array(d_fp_gen)
    print(f"\n[6] delta_current: mean={cur.mean():.4f} min={cur.min():.4f} max={cur.max():.4f}")
    print(f"    delta_forward: mean={fp.mean():.4f} min={fp.min():.4f} max={fp.max():.4f}")
    print(f"    max|delta_cur - delta_fwd| = {np.abs(cur - fp).max():.4f}")

if __name__ == "__main__":
    main()
