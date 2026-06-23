# TakeMeter: A Discourse-Quality Classifier for Retail Finance Communities

TakeMeter is a fine-tuned text classifier that sorts posts and comments from retail finance communities into four kinds of discourse: structured argument, confident assertion, question, and emotional reaction. The goal is to measure not what a post is *about*, but what it is *doing*, so that "is this real reasoning or just confidence?" becomes something a model can flag automatically.

This README is the final report. Design notes, edge-case rules, and working decisions live in [planning.md](planning.md).

---

## 1. Community

I chose **retail finance discussion subreddits**, primarily **r/stocks** and **r/CryptoCurrency**, with **r/investing** as an overflow source.

This community is a strong fit for a discourse-quality classifier because its central, constant tension is exactly the thing I want to measure: "is this researched, or is it just vibes?" Participants openly distinguish "DD" (due diligence) from "hopium," and they call out posts that predict without arguing. The discourse varies enormously in quality, from genuinely researched valuation arguments to one-line "to the moon" reactions, which gives a classifier real signal to learn. The text is public, abundant, and self-contained at the comment level, so collecting 200+ usable examples was practical. The distinction also matters to the people there: the whole point of these subs is separating signal from noise before risking money.

---

## 2. Label Taxonomy

Four labels, defined by the **structure** of a post rather than its quality. Structural distinctions are observable from the text alone (which is all the model sees) and two readers can agree on them.

### `analysis`
A claim supported by specific, checkable evidence (financials, valuation math, historical patterns, on-chain data, a named catalyst), or reasoning through a mechanism. The post argues: you could disagree by disputing its evidence.
- "DELL did over $95b in revenue in 2025 and are projecting ~$165b in 2026. A $1.44b multiyear contract is a drop in the bucket."
- "SpaceX closed near a $2.1 trillion market cap on 2025 revenue of $18.7 billion. That is a price-to-sales ratio of about 112x, and they posted a $4.9 billion net loss for the year."

### `speculation`
A confident directional call, or a bold opinion, asserted with little or no supporting evidence. The post predicts or asserts rather than argues. A single decorative statistic does not count as evidence.
- "SK Hynix has and will continue to outperform mu"
- "PL. Easy 2x by end of 2026."

### `question`
The post is primarily seeking information, advice, or input from the community. A post that lays out a personal dilemma and stalls on a decision counts as seeking input.
- "MU or SNDK which one looks better going forward?"
- "I bought MRVL at $82 off hiring data. It's $325 now and I can't decide whether to sell."

### `reaction`
An emotional response to a price move or market event (panic, euphoria, frustration, regret), with little or no argument.
- "Down 40% today because the Fed spooked everyone with the rate comments. Brutal."
- "I forgot I bought AMD and now its most of my portfolio, it's stressing me out"

---

## 3. Data Collection and Annotation

**Source.** Public posts and top-level comments from r/stocks and r/CryptoCurrency, collected in late June 2026 during a volatile market stretch (a hawkish Fed surprise, the SpaceX IPO aftermath, and Micron earnings week), which produced a rich mix of all four discourse types. Collection was mostly manual copy-paste into a spreadsheet, which kept me close to the data, with the option of a browser-based comment exporter for high-activity threads.

**Labeling process.** I used Claude to pre-label batches of posts against my planning.md definitions, then read and corrected every single pre-assigned label myself. Pre-labeling sped up throughput but did not replace review. I tracked which rows were pre-labeled and which I overrode in two spreadsheet columns (`prelabeled`, `changed_on_review`) so the assistance is fully disclosed (see AI Usage). I also applied filtering rules during collection, excluding news/headline relays, promotional or referral posts, bare statistic drops with no take, and reply fragments that could not stand alone without their parent comment.

**Label distribution (full dataset, 200 examples).**

| Label | Count | Share |
|-------|-------|-------|
| analysis | 70 | 35.0% |
| speculation | 46 | 23.0% |
| reaction | 43 | 21.5% |
| question | 41 | 20.5% |
| **Total** | **200** | **100%** |

No label exceeds the 70% imbalance threshold, and every label clears 20%. The notebook split this into 140 train / 30 validation / 30 test (70/15/15), stratified so all four labels appear in the test set.

**Three genuinely difficult examples and how I resolved them.**

1. **The one-stat flex (analysis vs speculation).** "ETH is massively undervalued, its P/E equivalent is way below its historical average." This cites a metric but makes a bold directional call. Decision rule: if the evidence would support the claim once you strip the opinion framing (specific, not cherry-picked), label `analysis`; if it is one decorative number selected for effect, label `speculation`. This one is a single metric with no comparison, so it goes to **`speculation`**.

2. **The question that argues (question vs analysis).** "Am I crazy for thinking AMD is undervalued? It's at a discount to NVDA on forward earnings despite taking MI300 share, and the data-center segment doubled YoY." The question mark is rhetorical; the body is a full argument the author already believes. Decision rule: label what the post is primarily doing. This is **`analysis`**.

3. **The reaction with a reason (reaction vs analysis).** "Down 40% today because the Fed spooked everyone with the rate comments. Brutal." A causal aside does not make a post an argument. The post is primarily venting, and "because the Fed" is a one-clause gesture, not developed reasoning, so it stays **`reaction`**.

---

## 4. Fine-Tuning Approach

**Base model.** `distilbert-base-uncased` from Hugging Face, fine-tuned with a four-class sequence-classification head on Google Colab's free T4 GPU.

**Training setup.** 140 training examples, tokenized, batch size 16, learning rate 2e-5.

**Hyperparameter decision (epochs).** This was the decision that mattered most, and I made it by responding to a failure. My first run used the default **3 epochs** and produced an accuracy of **0.467**, *worse than the zero-shot baseline*. The diagnosis was visible in two places: every prediction came back at roughly 0.26 confidence (near the 0.25 random-chance level for four classes), and the confusion matrix showed the model only ever predicting two of the four classes, never `speculation` or `reaction`. That is the signature of an undertrained model that has barely moved off its random initialization, not a data problem (the dataset and splits were already verified as balanced, and the baseline proved the task is learnable). I increased training to **10 epochs** and re-ran. Accuracy rose to **0.767**, confidence scores lifted off chance, and the model began predicting all four classes. Training took only a few minutes on the T4 at either setting, so more epochs cost almost nothing.

---

## 5. Baseline

The baseline is a **zero-shot prompt to Groq's `llama-3.3-70b-versatile`**, classifying each test example with no task-specific training. This tells me how hard the task is for a strong general model, which gives the fine-tuned model's performance real meaning.

**Prompt.** The system prompt named the community and task, defined all four labels in the same language as planning.md, gave one real example post per label, included the same hard-case decision rules I labeled with, and instructed the model to output only the lowercase label name. Constraining the output format this way mattered: the notebook does exact string matching, so a reply like "Analysis." would be marked unparseable.

**How results were collected.** The prompt ran against all 30 examples in the locked test set, before any fine-tuning. All 30 responses parsed cleanly (0% unparseable), so no prompt revision was needed. Both models were evaluated on the identical test set.

---

## 6. Evaluation Report

### Overall accuracy

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Zero-shot baseline (Groq llama-3.3-70b) | 0.700 | 0.72 |
| Fine-tuned DistilBERT (10 epochs) | **0.767** | **0.73** |
| Improvement | **+0.067** | +0.01 |

### Per-class metrics

**Baseline (Groq, zero-shot):**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| speculation | 0.67 | 0.86 | 0.75 | 7 |
| analysis | 1.00 | 0.45 | 0.62 | 11 |
| reaction | 0.45 | 0.83 | 0.59 | 6 |
| question | 1.00 | 0.83 | 0.91 | 6 |

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| speculation | 1.00 | 0.29 | 0.44 | 7 |
| analysis | 0.69 | 1.00 | 0.81 | 11 |
| reaction | 0.80 | 0.67 | 0.73 | 6 |
| question | 0.86 | 1.00 | 0.92 | 6 |

### Confusion matrix (fine-tuned model)

Rows are true labels, columns are predicted labels.

| True \ Predicted | speculation | analysis | reaction | question |
|------------------|:-----------:|:--------:|:--------:|:--------:|
| **speculation** | 2 | 4 | 1 | 0 |
| **analysis** | 0 | 11 | 0 | 0 |
| **reaction** | 0 | 1 | 4 | 1 |
| **question** | 0 | 0 | 0 | 6 |

The committed image version is [`confusion_matrix.png`](confusion_matrix.png).

The diagonal (correct predictions) holds for analysis (11/11) and question (6/6). The dominant off-diagonal cell is **speculation predicted as analysis (4 of 7 speculation posts)**. This single cell accounts for most of the model's errors and points directly at the boundary the model has not fully learned.

### Three wrong predictions, analyzed

The model made 7 errors total. The three below are chosen because they share the dominant pattern and because the model's confidence on them is revealing.

**Wrong prediction 1 (speculation predicted as analysis, confidence 0.82).**
Post: "It is possible that in 2028 we may see a glut of memory supply, and these stocks may come down or be stagnant for next 3-4 years, but you might have made your profit. Or the downturns might not be severe."
Why it failed: this is a hedged forward scenario with a gestured mechanism (supply glut leads to lower prices). The reasoning is one clause, not a developed argument, but it has the surface shape of analysis (a cause, a consequence, a timeframe), and the model bought the form. The high confidence (0.82) is the worrying part: the model is not unsure here, it is confidently wrong, which means it has genuinely encoded "cause-and-effect phrasing equals analysis" rather than weighing whether the evidence is real.

**Wrong prediction 2 (speculation predicted as analysis, confidence 0.79).**
Post: "Markets have gone from irrational to outright degenerate. There is absolutely no rational reason why this company should be worth anywhere near $3T, lol."
Why it failed: this is an undefended bearish assertion with no figures and no comparison, the mirror image of a real valuation argument. The model most likely keyed on the financial vocabulary ("worth," "$3T," "rational") and the confident tone, and missed that no evidence is actually offered. Again the confidence is high (0.79), so this is a systematic blind spot, not a coin-flip.

**Wrong prediction 3 (speculation predicted as analysis, confidence 0.78).**
Post: "It will go up after earnings. As soon as the earnings news comes out, it might drop a few % because people will sell, but it will go up soon after, just like Sandisk in the last earnings call."
Why it failed: a directional call whose only support is a thin analogy ("just like Sandisk"). The model read the presence of a comparison as evidence of an argument. It learned to detect evidence-shaped language but not whether the evidence supports the claim.

**The pattern.** Four of the seven errors are speculation predicted as analysis, and three of those four came with confidence between 0.78 and 0.82. The errors are not the model hesitating at a hard boundary; they are the model confidently applying a rule it learned a little too well: posts that *mention* evidence or use cause-and-effect and financial language get classified as `analysis`, whether or not the evidence is real. This is consistent with the confusion matrix (4 of 7 speculation posts went to analysis) and the per-class metrics (analysis recall is a perfect 1.00 but precision falls to 0.69, meaning the label is over-applied). The remaining errors are more excusable near-ties: "nothing will beat the highs of crypto vs SPY 0DTE" was called reaction at only 0.45, the "power law floor" price-forecast post was called analysis at 0.51, the "Micron is soaring" hype blurb went to analysis (a reaction the model read as reporting), and "If you know Bitcoin is gonna crash, why aren't you all millionaires already?" (a rhetorical reaction) was read as a literal question. The analysis-to-speculation distinction I flagged as hardest from the start is still the model's weak point; it just errs on the opposite side from the baseline now.

### Sample classifications

These five posts were run through the fine-tuned model. The confidence values for the two misclassified posts are taken from the Section 4 wrong-predictions output. The three correct predictions below are marked so; their exact confidence scores can be read off the deployed interface (Section 4 only prints confidence for wrong predictions).

| Post (truncated) | True label | Predicted | Confidence | Correct? |
|------------------|-----------|-----------|------------|----------|
| "DELL did over $95b in revenue in 2025... a $1.44b contract is a drop in the bucket." | analysis | analysis | [from interface] | Yes |
| "MU or SNDK which one looks better going forward?" | question | question | [from interface] | Yes |
| "Down 40% today because the Fed spooked everyone. Brutal." | reaction | reaction | [from interface] | Yes |
| "Markets have gone from irrational to outright degenerate... no rational reason worth $3T, lol." | speculation | analysis | 0.79 | No |
| "It is possible that in 2028 we may see a glut of memory supply..." | speculation | analysis | 0.82 | No |

**Why one correct prediction is reasonable.** "DELL did over $95b in revenue in 2025 and are projecting ~$165b in 2026. A $1.44b multiyear contract is a drop in the bucket." is correctly labeled `analysis`. The prediction is reasonable because the post does exactly what the label requires: it cites specific figures and reasons from them to a conclusion (the contract is immaterial relative to revenue). There is a checkable argument here, not just an assertion, so the model's call lines up with the label definition. Notably, this is a case where real numbers back a real conclusion, which is precisely the case the model handles well; its failures come from posts that have the *look* of this without the substance.

---

## 7. Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the difference between *arguing* and *asserting*: whether a post backs its claim with real, checkable evidence. What it actually learned is a near neighbor of that, but not the same thing: it learned to detect **the surface features of evidence**, financial vocabulary, numbers, comparisons, cause-and-effect phrasing, and to treat their presence as analysis.

The clearest sign is the asymmetry in the analysis class: recall 1.00, precision 0.69. The model catches every real analysis post, but it also sweeps in confident posts that merely *sound* analytical (the one-stat flex, the hedged scenario, the undefended valuation claim with a dollar figure in it). It overfit to the *form* of evidence rather than the *substance* of it. That is exactly the distinction my taxonomy was built around, and it is the part a 200-example dataset apparently could not fully teach.

What it learned well: `question` (F1 0.92), which has strong surface cues, and `analysis` recognition in general. What it missed: the speculation boundary, where telling decorative evidence from real evidence requires a judgment the model did not acquire. Interestingly, the baseline failed the same boundary in the opposite direction (it under-predicted analysis), which suggests the analysis-to-speculation distinction is the genuinely hard core of this task for both a general model and a fine-tuned one.

---

## 8. Spec Reflection

**One way the spec helped.** The spec's insistence on writing label definitions and edge-case decision rules in planning.md *before* collecting any data shaped everything downstream. Forcing the one-stat-flex and question-that-argues rules into writing up front meant I annotated consistently, and, unexpectedly, those same edge cases turned out to be exactly where the model failed. The design discipline gave me a ready-made hypothesis to test against the results.

**One way my implementation diverged.** The spec frames the hyperparameter choice as a decision you make and document. In practice I did not choose the final value deliberately at the start; I diverged by having to *debug* it. My first run at the default 3 epochs underperformed the baseline, and only by diagnosing the failure (near-chance confidence, two-class collapse) did I arrive at 10 epochs. The divergence was that the "decision" was reactive rather than planned, which in hindsight produced a more honest and better-understood result than picking a number blind.

---

## 9. AI Usage

**Instance 1: Annotation pre-labeling (disclosed).** I directed Claude to pre-label batches of unlabeled posts against my planning.md label definitions and decision rules, outputting one label per post. I then reviewed every pre-assigned label myself and overrode the ones I disagreed with, tracking pre-labeled and overridden rows in the `prelabeled` and `changed_on_review` columns of my dataset. Claude's pre-labels were a starting point; the final labels are my own reviewed judgments. The borderline cases (especially analysis vs speculation) were where I overrode most often.

**Instance 2: Failure-pattern analysis.** After fine-tuning, I gave Claude the list of misclassified test examples and asked it to identify common themes. It surfaced the speculation-to-analysis direction as the dominant pattern and the "evidence-shaped but not evidence-backed" hypothesis. I verified this myself by re-reading each misclassified post and checking it against the confusion matrix and per-class precision/recall before writing the analysis above. I kept the patterns I could confirm in the data and discarded anything I could not.

**Instance 3: Label stress-testing (design phase).** Before annotating, I asked Claude to generate posts sitting on the analysis-speculation boundary to test whether my definitions held up. Where the generated posts were hard to classify cleanly, I tightened the decision rules in planning.md.

---

## Repository Contents

- `planning.md` - design thinking, label definitions, edge-case rules, data and evaluation plans
- `data.csv` - the full labeled dataset (200 examples)
- `data_prelabel.csv` - the full pre-labeled/changed dataset from claude and I (200 examples)
- `confusion_matrix.png` - fine-tuned model confusion matrix
- `evaluation_results.json` - baseline vs fine-tuned metrics summary
- `README.md` - this report
