# TakeMeter - Planning

**Project:** AI201 Project 3 - TakeMeter (fine-tuned discourse-quality classifier)
**Community:** r/stocks / r/CryptoCurrency (retail finance discussion)
**Task:** Classify a post by *what kind of discourse it is* - argument, prediction, question, or emotional reaction.

---

## 1. Community

I chose **retail finance discussion subreddits** - primarily **r/stocks** and **r/CryptoCurrency**, with r/investing as an overflow source if a label runs thin.

I picked this community because it's where I actually spend time (crypto, stocks, ETFs), so I understand the norms and can label consistently from genuine familiarity rather than guessing. More importantly, it's an unusually good fit for a *discourse-quality* classifier: the central, constant tension in these communities is **"is this real reasoning, or just confidence?"** People openly distinguish "DD" (due diligence) from "hopium," and call out posts that predict without arguing. That tension is exactly what the project asks me to measure, and it produces text that varies enormously in quality - from genuinely researched valuation arguments to one-line "to the moon" reactions. The discourse is text-heavy, public, and abundant, so collecting 200+ examples is straightforward.

The distinction matters to participants because the whole point of these subs is to separate signal from noise before risking money. A regular can tell the difference between a researched thesis and a hype-based price call instantly. That intuition is what I'm trying to teach the model.

---

## 2. Labels

### `analysis`
A claim supported by specific, checkable evidence - financials, valuation math, on-chain data, macro reasoning, or a named catalyst. **The post argues:** you could disagree with it by disputing its evidence.

- *Example 1:* "NVDA's forward P/E is ~30 vs a 5-yr average of 45, and data-center revenue grew 200% YoY last quarter. If that growth holds even halfway, current price isn't stretched relative to history."
- *Example 2:* "VOO and VTI overlap ~85% by holdings, so holding both doesn't diversify you, it just doubles your large-cap US exposure"

### `speculation`
A confident directional call ("X is going to $500," "this is the bottom") asserted with little or no supporting evidence. **The post predicts but doesn't argue** - the claim might be right, but it's an assertion, not a case.

- *Example 1:* "I see DRAM pushing to $100 and MU hitting $1500 by Q4."
- *Example 2:* "Bitcoin Could Reach $500k by Fifth Halving (2028) Due to Dual Demand"

### `question`
The post is **primarily seeking information or advice** - asking, not telling.

- *Example 1:* "Is it worth holding VOO and VTI together or is that redundant overlap?"
- *Example 2:* "The Iran war is supposedly over, Iran accepted it and the U.S. accepted it too, but why are the markets still weak? Why is Bitcoin still falling?"

### `reaction`
An **immediate emotional response** to a price move or event - panic, euphoria, frustration. Little to no argument.

- *Example 1:* "Down 40% on my portfolio today. I'm so cooked. Why do I do this to myself."
- *Example 2:* "WE ARE SO BACK. Green day, finally. Feels good to be alive."

---

## 3. Hard Edge Cases

### Primary edge case: the one-stat flex (`analysis` vs `speculation`)

A post makes a bold directional call but drops in **one** number to sound credible.

> *Example:* "ETH is massively undervalued - its P/E equivalent is way below its historical average."

Is that `analysis` (cites a metric) or `speculation` (bold directional call)?

**Decision rule:** If the evidence would actually support the claim once you strip the opinion framing - real, specific, not cherry-picked - label `analysis`. If it's one decorative number selected for effect, just enough to sound credible but not genuinely reasoning, label `speculation`. The example above leans **`speculation`**: a single metric, no comparison or mechanism, framing carries the weight.

### Secondary edge case: the question that argues (`question` vs `analysis`)

A post asks something but then lays out a full thesis underneath it.

> *Example:* "Am I crazy for thinking AMD is undervalued here? It's trading at a discount to NVDA on forward earnings despite taking MI300 market share, and the data-center segment doubled YoY…"

**Decision rule:** Label by **what the post is primarily doing.** If the question is a rhetorical frame around an argument the author already believes and supports, it's `analysis`. If the author genuinely doesn't know and is seeking input, it's `question`. The example above is `analysis` - the "am I crazy" is a hook, not a real request for information.

### Tertiary edge case: reaction with a reason (`reaction` vs `analysis`)

> *Example:* "Down 40% today because the Fed spooked everyone with the rate comments. Brutal."

**Decision rule:** A causal aside doesn't make it analysis. If the post is primarily venting emotion and the "reason" is a one-clause gesture (not a developed argument), it stays `reaction`. Only promote to `analysis` if the post genuinely reasons through the cause.

---

## 4. Data Collection Plan

**Source:** Public posts and top-level comments from r/stocks and r/CryptoCurrency (r/investing as overflow). Public only - no authenticated or private content.

**Method:** Manual copy-paste into a spreadsheet. Manual collection keeps me close to the data and is fast enough at this scale. I'll pull from a mix of daily discussion threads (rich in `reaction` and `question`), dedicated DD/analysis posts (`analysis`), and price-move threads (`speculation`, `reaction`) so I don't accidentally over-sample one label by only reading one kind of thread.

**Target distribution:** ~50 per label for a balanced ~200. I will try to keep it an even split making sure each label encompass ~20%. I expect `analysis` to be the scarcest (good arguments are rarer than reactions), so I'll over-collect those deliberately.

**If a label is underrepresented after 200:** Go back to label-specific sources - DD-flaired posts for `analysis`, sharp green/red days for `reaction`, "what should I buy" threads for `question` - and collect more of just that label until no label exceeds ~70% and each clears ~20%. I will **not** down-sample the majority class to force balance (that throws away real data). I'll up-sample the minority by collecting more.

**Filtering:** Skip image-only / screenshot posts with no meaningful text, deleted/removed posts, and posts that only make sense with external thread context or in-jokes the model can't see.

**Storage:** Single CSV with columns `text`, `label`, and `notes` (for difficult-case annotations). One complete file - the notebook handles the 70/15/15 split.

---

## 5. Evaluation Metrics

**Accuracy alone is not enough** because even with a roughly balanced 4-class set, accuracy hides *which* distinctions the model learned. A model could score decently on overall accuracy while completely failing on `analysis` (my scarcest and most important class) and I'd never see it from one number.

Metrics I'll use and why:

- **Overall accuracy** - headline number, and the direct point of comparison against the zero-shot Groq baseline. Tells me whether fine-tuning helped at all.
- **Per-class precision, recall, and F1** - the core of the evaluation. The class I most care about is `analysis`; I specifically want its **recall** (am I catching the genuinely well-reasoned posts?) and its **precision** (am I mislabeling confident-but-empty `speculation` as `analysis`?). The `analysis`, `speculation` boundary is the hard, meaningful one, so its per-class F1 is the real measure of success.
- **Macro-averaged F1** - averages F1 across classes equally, so a strong majority class can't paper over a weak minority class. This is my single summary metric, more honest than accuracy under any residual imbalance.
- **Confusion matrix** - to see *direction* of errors. I expect most errors to cluster at `analysis` and `speculation`; the matrix confirms whether that's true and which way the model leans. Directional error patterns are what I'll dig into for the failure analysis.

---

## 6. Definition of Success

For this to be genuinely useful as a "is this real reasoning?" filter in a real community tool, I'd want:

- **Macro F1 ≥ 0.70** on the fine-tuned model - meaning it learned all four distinctions reasonably, not just the easy ones.
- **`analysis` per-class F1 ≥ 0.65** specifically - this is the class the whole tool exists to identify, so it's the binding constraint. A high overall score with weak `analysis` is a failure for my use case.
- **Fine-tuned model beats the zero-shot baseline on macro F1 by a clear margin** (not within noise). If fine-tuning barely beats a 70B model with no training, that tells me my labels are too easy or too noisy - and that's a finding worth reporting honestly either way.

**Realistic "good enough" for deployment:** I'd accept macro F1 in the **0.65–0.75** range with `analysis` F1 ≥ 0.65 as a usable first-pass filter. Something that surfaces likely-reasoned posts for a human to confirm, not an autonomous judge. This is a hard, subjective task on 200 examples; >0.95 accuracy would make me suspect label leakage or labels that are too easy, and I'd investigate rather than celebrate.

---

## 7. AI Tool Plan

### Label stress-testing (before annotating)
I'll give Claude my four label definitions and the edge-case decision rules above and ask it to generate 5–10 posts that sit on the `analysis`, `speculation` boundary. If I can't cleanly classify what it produces using my own rules, the definitions need tightening. I'll fix them *before* annotating 200 examples, not after.

### Annotation assistance (during collection)
I'll use Claude for selective pre-labeling, then review every label myself. I'll feed Claude batches of unlabeled posts plus my `planning.md` label definitions and decision rules, and have it assign exactly one label per post. Then I'll read every post myself and correct any label I disagree with - pre-labeling with Claude speeds throughput but does **not** replace genuine review (skimming produces noisy training data).

**How I'll track pre-labeled rows:** I'll add a `prelabeled` column to my CSV (`yes`/`no`) marking which rows Claude pre-labeled, plus a `changed_on_review` column (`yes`/`no`) flagging the ones where my review overrode Claude's label. This lets me (1) disclose the exact pre-labeled set in the README's AI usage section, and (2) report how often I had to correct Claude, which is a useful signal about how reliable the pre-labeling actually was. These two tracking columns are for my own records and disclosure - I'll drop them before uploading the CSV to the notebook, which only needs `text` and `label`.

### Failure analysis (during evaluation)
After fine-tuning, I'll paste the list of misclassified test examples into an LLM and ask it to surface patterns - recurring label pairs, post length, sarcasm, low-information posts. Then I'll verify each proposed pattern by re-reading the actual examples myself before writing it up. The AI surfaces candidates; I confirm or discard them, and I'll report what I had to correct.

