"""Step 1: token alignment 検証
- 93 vs 94 のズレの位置を特定する
- 現行のspan決定ロジックの挙動を再現する
- overlap条件で取り直したspanが "William Shakespeare" 全体を指すことを確認する
"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from spilled_energy.generation import generate_answer

MODEL = "meta-llama/Meta-Llama-3-8B"
QUESTION = "Who wrote the play Romeo and Juliet?"
ANSWER_STR = "William Shakespeare"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    prompt = f"Q: {QUESTION}\nA:"
    out = generate_answer(
        prompt=prompt,
        model=model,
        tokenizer=tok,
        max_new_tokens=100,
        do_sample=False,
        device="cuda",
    )

    text = out["text"]
    seqs = out["sequences"]
    n_steps = len(out["scores"])
    input_len = seqs.shape[1] - n_steps
    gen_ids = seqs[0, input_len:].tolist()

    print(f"[1] generated_text (repr): {text!r}")
    print(f"[2] n_generated={len(gen_ids)}, last_id={gen_ids[-1]}, eos_id={tok.eos_token_id}")

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    re_ids = enc.input_ids
    print(f"[3] retokenized={len(re_ids)}, generated={len(gen_ids)}")

    n = min(len(re_ids), len(gen_ids))
    first_diff = next((i for i in range(n) if re_ids[i] != gen_ids[i]), None)
    print(f"[4] first mismatch index: {first_diff}")
    if first_diff is None and len(gen_ids) > len(re_ids):
        extra = gen_ids[len(re_ids):]
        print(f"    -> 差分は末尾のみ: {extra} = {tok.convert_ids_to_tokens(extra)}")

    start_idx = text.find(ANSWER_STR)
    end_idx = start_idx + len(ANSWER_STR)
    print(f"[5] char span: {start_idx}-{end_idx}")

    offsets = enc.offset_mapping

    ts_old = te_old = None
    for i, (s, e) in enumerate(offsets):
        if s >= start_idx and ts_old is None:
            ts_old = i
        if s < end_idx:
            te_old = i + 1

    print(f"[6] 現行span: {ts_old}-{te_old} -> {tok.decode(re_ids[ts_old:te_old])!r}")

    ts_new = te_new = None
    for i, (s, e) in enumerate(offsets):
        if e > start_idx and s < end_idx:
            if ts_new is None:
                ts_new = i
            te_new = i + 1

    print(f"[7] 修正span : {ts_new}-{te_new} -> {tok.decode(re_ids[ts_new:te_new])!r}")

    print(f"[8] generated_ids側 decode: {tok.decode(gen_ids[ts_new:te_new])!r}")

if __name__ == "__main__":
    main()
