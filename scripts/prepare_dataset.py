#!/usr/bin/env python3
"""
Convert customer's red_hat_sample.jsonl into an OpenAI-chat-format JSONL that
mm_bench.py can replay, and print the dataset stats we need for the baseline.

Input row format (from customer):
  {"item_id": ..., "messages": [...], "image_base64": "..."}
  where messages[1].content has [{"type":"image"}, {"type":"text","text":...}]

Output row format:
  {"item_id": ..., "messages": [<openai chat messages with image_url data URI>]}

Usage:
  python3 prepare_dataset.py red_hat_sample.jsonl customer_attr.jsonl \
      [--size 480] [--repeat 1000] [--tokenizer google/gemma-4-31B-it]

--repeat N cycles the rows to reach N requests (noted in the stats output so
  it ends up in the results, since cycling inflates prefix/image cache hits).
--tokenizer is optional; if transformers isn't installed you get a char-based
  estimate clearly labelled as such.
"""
import argparse, base64, io, json, statistics, sys
from PIL import Image


def to_data_uri(b64: str, size: int | None) -> tuple[str, tuple[int, int], tuple[int, int]]:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    orig = img.size
    if size and img.size != (size, size):
        img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    out = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{out}", orig, img.size




def perturb_data_uri(b64: str, size: int | None, seed: int) -> str:
    """Re-encode with a small deterministic pixel stamp so image bytes are unique
    per request. Defeats image-embedding / full-prefix caching without meaningfully
    changing content (stamp is a 6x6 corner block)."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    px = img.load()
    r, g, b = (seed * 37) % 256, (seed * 101) % 256, (seed * 197) % 256
    for x in range(6):
        for y in range(6):
            px[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def convert_messages(msgs, data_uri):
    out = []
    for m in msgs:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for p in content:
            if p.get("type") == "image":
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif p.get("type") == "text":
                parts.append({"type": "text", "text": p["text"]})
            else:
                parts.append(p)
        out.append({"role": m["role"], "content": parts})
    return out


def text_of(msgs, role):
    for m in msgs:
        if m["role"] == role:
            c = m["content"]
            if isinstance(c, str):
                return c
            return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=480, help="resize images to NxN (0 = keep as-is)")
    ap.add_argument("--repeat", type=int, default=0, help="cycle rows to reach this many requests")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id for real token counts")
    ap.add_argument("--unique", action="store_true", help="when cycling, perturb image+text per copy so only the system prompt is cache-shareable (production-like)")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[warn] tokenizer load failed ({e}); falling back to char estimate", file=sys.stderr)

    def ntok(s):
        if tok:
            return len(tok.encode(s))
        return round(len(s) / 4)  # rough estimate only

    rows, sys_prompts, sys_toks, usr_toks, orig_sizes = [], set(), [], [], set()
    with open(args.src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uri, orig, new = to_data_uri(r["image_base64"], args.size or None)
            orig_sizes.add(orig)
            msgs = convert_messages(r["messages"], uri)
            sp, up = text_of(r["messages"], "system"), text_of(r["messages"], "user")
            sys_prompts.add(sp)
            sys_toks.append(ntok(sp))
            usr_toks.append(ntok(up))
            rows.append({"item_id": r.get("item_id"), "messages": msgs, "_b64": r["image_base64"]})

    n_real = len(rows)
    if args.repeat and args.repeat > n_real:
        if args.unique:
            out_rows = []
            for i in range(args.repeat):
                base = rows[i % n_real]
                if i < n_real:
                    out_rows.append(base)
                    continue
                uri = perturb_data_uri(base["_b64"], args.size or None, i)
                msgs = json.loads(json.dumps(base["messages"]))  # deep copy
                for m in msgs:
                    if m["role"] != "user" or isinstance(m["content"], str):
                        continue
                    for p in m["content"]:
                        if p.get("type") == "image_url":
                            p["image_url"]["url"] = uri
                        elif p.get("type") == "text":
                            p["text"] = f'[request-{i}] ' + p["text"]
                out_rows.append({"item_id": f'{base["item_id"]}-u{i}', "messages": msgs, "_b64": None})
            rows = out_rows
        else:
            rows = [rows[i % n_real] for i in range(args.repeat)]

    with open(args.dst, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_b64"}) + "\n")

    unit = "tokens" if tok else "tokens (CHAR ESTIMATE, len/4 — install transformers for real counts)"
    print(f"rows (real):            {n_real}")
    print(f"rows (written):         {len(rows)}{'  <- cycled' if len(rows) > n_real else ''}")
    print(f"distinct system prompts:{len(sys_prompts)}")
    print(f"system prompt {unit}: mean={statistics.mean(sys_toks):.0f} min={min(sys_toks)} max={max(sys_toks)}")
    print(f"user prompt   {unit}: mean={statistics.mean(usr_toks):.0f} min={min(usr_toks)} max={max(usr_toks)}")
    print(f"text-only input {unit}: mean={statistics.mean(sys_toks) + statistics.mean(usr_toks):.0f}  (image tokens NOT included)")
    print(f"original image sizes:   {sorted(orig_sizes)}")
    print(f"output image size:      {args.size}x{args.size}" if args.size else "output image size:      unchanged")
    print(f"images per request:     1")
    print()
    print("NOTE: image token count is model-specific and not included above.")
    print("      Check it from the server's usage.prompt_tokens on a single request.")


if __name__ == "__main__":
    main()
