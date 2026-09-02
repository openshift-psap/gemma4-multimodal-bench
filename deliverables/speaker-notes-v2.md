# Speaker Notes — Customer Presentation (plain version)

Audience: customer team. Tone: we ran your workload, here's what we found,
here's what we want to try next. Simple words. Short sentences.
Say "your data", "your team told us" — this is their workload, we're
reporting back.

---

## Opening

"Thanks for sharing the dataset and the baseline numbers. We ran both
models on your real data — your prompts, your images — on a single H100.
This is what we measured. Today is about the data, not conclusions."

If asked about setup:
- One H100 80GB, vLLM, single GPU as per the criteria doc
- Your 100 sample requests, expanded to 1000 for sustained load
- About 3.2K input tokens per request with one image, 512 output tokens
- We tested at 1, 16, and 64 concurrent requests

---

## The two cards

"Good news first: with default settings, both models already beat the
baseline numbers. 26B gave us 3.1 QPS. 31B gave us 0.88."

"But the two models behave very differently, and that's the main thing
we learned. The 26B is limited by compute. The 31B is limited by memory.
So they need different fixes."

Note on baselines: your team shared updated numbers with us — 1.5 for
the 26B and 0.8 for the 31B. If the 2x target follows the new baselines,
the 26B is already there. Good to confirm that together in this meeting.

---

## The memory picture (the two bars)

"This one picture explains most of the report. Same GPU, same 80GB.
The 26B fits about 198K tokens of cache — around 60 requests at once.
The 31B fits only 43K — about 13 requests. After that, requests wait
in line."

"That waiting is why the 31B showed 43 second first-token times at high
load. When we turned on FP8 cache, capacity doubled, waiting dropped to
7.5 seconds, and throughput went up 21%."

---

## The charts

Chart 1 (26B): "The three config lines sit on top of each other. Server
settings did not move this model at all. The dotted line is the same
setup when items repeat — throughput doubles to 6.1. So how often your
items repeat matters a lot here. Your team told us images repeat 60-70%
but items usually change — we're building a test that matches that mix."

Chart 2 (31B): "Only one setting moved this model — FP8 cache. Because
its problem is memory, not compute."

Chart 3: "This is the waiting time disappearing when cache capacity
doubled. 43 seconds down to 7.5."

---

## The prompt finding — say this carefully

"One thing we noticed in the prompt structure. The rules section — about
2K tokens, same in every request — comes after the image. Caching works
front to back, and every image is different, so those rules get
recomputed every single time."

"The prompt is written for model quality, and that's the right priority.
But if we move the rules before the image — same text, just reordered —
they get cached once and reused. We think this is the biggest untested
improvement for the 26B. We'd measure throughput on our side, and your
team would check accuracy doesn't change. Nothing ships without that."

---

## What we want to run next

"Three things, in order:
1. The reordered prompt — measure the caching gain
2. Multi-image requests — your team said 3-7 images for moderation, so
   we'll test 1, 5, and 9 images per request using your exact format
3. The 31B on a 4-bit checkpoint — smaller weights means more cache,
   which is exactly what that model needs"

---

## Questions they might ask

**"So did you hit 2x?"**
"Against the updated baselines — the 26B looks like yes, 3.1 vs a 3.0
target, and we haven't even applied the prompt change yet. The 31B is at
1.06 of 1.6. We have a clear path for it but we won't claim it before we
measure it."

**"We saw 6.1 somewhere — what's that?"**
"That's the 26B when items repeat and get served from cache. All-new
items give 3.0. Your production is somewhere in between, and your
repeat-rate answer helps us test the right mix. We show both numbers so
nobody gets surprised."

**"Your 31B number at concurrency 16 is lower than ours." (0.68 vs 0.8)**
"Two likely reasons. Your GPUs have 94GB, ours have 80 — and this model
is memory-bound, so that gap matters. And your test was 100 requests,
ours ran sustained for minutes, which fills the queue more. We're
rerunning with your exact method — concurrency 16, 100 requests, 5
warmup — so we compare the same thing."

**"Is the prompt change safe?"**
"Honestly — we don't know yet, and we won't pretend to. Reordering can
change outputs. We measure speed, your team checks accuracy against
ground truth. If accuracy holds, it's a free win. If not, we drop it."

**"Why one image per request in your tests?"**
"That's what the sample dataset had, so we started there. Now that your
team clarified 3-7 images for moderation and shared the format — images
after the product text — the multi-image runs are next, matching your
structure exactly."

**"You said around 1700 tokens per image, your numbers look smaller?"**
"Right — worth clearing up. On the vLLM Gemma build we measured roughly
100-300 tokens per image. The 1700 figure may be from the Gemini side —
different tokenizer. Let's confirm which applies to your deployment,
because it changes the math."

**"Can we run this ourselves?"**
"Yes. Everything is scripts plus your own data format. We hand over the
whole thing — converter, load generator, results. You should be able to
reproduce every number in this report."

**"What about accuracy?"**
"We deliberately didn't touch it. Per the plan: we deliver recipes and
throughput data, your team validates accuracy against ground truth. Any
recipe that drops accuracy goes back to us for another pass."

---

## If you only get to say three things

1. "Both models beat the baselines with default settings — the gap to
   2x is smaller than expected."
2. "The two models need different fixes: the 26B needs the prompt
   reorder, the 31B needs more memory for cache."
3. "Next runs: reordered prompt, multi-image, 4-bit 31B. All measured,
   all reproducible, accuracy always validated by your team."
