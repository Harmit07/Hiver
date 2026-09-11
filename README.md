# 🤖 AI Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment** — An AI customer support agent that classifies intents, drafts grounded replies, and makes escalation decisions for Apple Support on Twitter.

## ⚡ Quick Start (Reproduce Results in < 15 Minutes)

### Prerequisites
- Python 3.10+
- OpenRouter API key ([get one here](https://openrouter.ai/))
- The Twitter Customer Support dataset (`twcs.csv`) — [Kaggle link](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd Hiver
pip install -r requirements.txt

# Place dataset
mkdir -p data/raw
# Download twcs.csv from Kaggle and place it in data/raw/
# Or: kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/

# Set API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Run full pipeline
python run_pipeline.py
```

The pipeline caches intermediate results — if you've already run a step, it skips it on re-run. This means you can re-run `python run_pipeline.py` at any time without re-spending API credits.

### What the pipeline does:
1. **Preprocess** — Filters 2.8M tweets to AppleSupport, reconstructs 81K conversation threads, subsamples 5K
2. **Discover intents** — Embeds messages, clusters with K-Means, uses LLM to label clusters into 8-12 intents
3. **Build golden set** — Stratified sample of 200 examples for evaluation
4. **Run agent** — Classifies intents, drafts replies, makes escalation decisions on all golden set examples
5. **Run baselines** — Trivial (majority class) and Simple (TF-IDF + LogReg) baselines
6. **LLM judge** — Claude Sonnet evaluates reply quality on 5 dimensions
7. **Generate reports** — Metrics, confusion matrices, comparison tables

### Useful flags:
```bash
python run_pipeline.py --force      # Run everything from scratch (ignores cached data)
python run_pipeline.py --skip-llm   # Skip LLM-dependent steps (uses cached results)
python run_pipeline.py --step 4     # Start from step 4 (agent evaluation)
```

---

## 📐 Architecture

```
Customer Message
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│   Intent     │────▶│   Retrieval      │
│  Classifier  │     │  (FAISS, top-5   │
│  (LLM few-   │     │   similar        │
│   shot)      │     │   threads)       │
└──────────────┘     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Reply Drafter   │
                     │  (LLM, grounded  │
                     │   in history)    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Escalation     │
                     │   Decision       │
                     │  (Rules + LLM)   │
                     └──────────────────┘
```

### Components

| Component | Model/Method | Purpose |
|-----------|-------------|---------|
| Intent Classifier | Gemini Flash (few-shot) | Classify into data-driven taxonomy |
| Retriever | FAISS + all-MiniLM-L6-v2 | Find similar historical threads |
| Reply Drafter | Gemini Flash + retrieved context | Generate grounded, on-brand replies |
| Escalation Engine | Rule-based + LLM hybrid | Decide auto-handle vs. human |
| LLM Judge | Claude Sonnet (different model!) | Evaluate reply quality |

**Key design choice**: Using *different models* for generation (Gemini Flash) and evaluation (Claude Sonnet) avoids self-evaluation bias.

---

## 📊 Results

Full results are generated in `results/evaluation_summary.md` after running the pipeline.

### Intent Classification

| System | Accuracy | Macro F1 |
|--------|----------|----------|
| **LLM Agent (ours)** | **79.5%** | **0.835** |
| Simple (TF-IDF + LogReg) | 59.0% | 0.605 |
| Trivial (majority class) | 12.5% | 0.050 |

### Reply Quality (LLM-as-Judge, 1-5 scale)

*Agent overall score: **4.52/5.0***

| Dimension | Description |
|-----------|-------------|
| Relevance | Does the reply address the actual issue? |
| Tone | Professional, empathetic, on-brand? |
| Actionability | Concrete next steps? |
| Grounding | Consistent with how Apple actually responds? |
| Safety | No hallucinated policies or promises? |

---

## 🔬 Evaluation Deep Dive

### Golden Evaluation Set (200 examples)
- **Stratified by intent**: Every intent category is represented
- **Stratified by difficulty**: Easy (40%), Medium (30%), Hard (30%)
- **Adversarial examples**: Sarcasm, off-topic, multi-language (~20 examples)
- See `evaluation/sampling_methodology.md` for full methodology

### Baselines
1. **Trivial**: Always predicts majority intent, returns template reply, never escalates
2. **Simple**: TF-IDF + Logistic Regression for intent, BM25 retrieval for reply, keyword escalation

### LLM-as-Judge Calibration
We measure agreement between the LLM judge (Claude Sonnet) and human labels on 50 examples using:
- Pearson correlation (r)
- Cohen's kappa (κ)
- Mean Absolute Error (MAE)

This proves the judge is trustworthy — or honestly shows where it isn't.

---

## ⚠️ What is Misleading About My Headline Number?

1. **Self-labeling bias** — We defined the intent categories AND labeled the golden set, creating circular reasoning
2. **Cherry-picked difficulty** — Our "hard" examples are hard by *our heuristics*, not by real-world standards
3. **LLM judge is not ground truth** — 50-example calibration is small; the judge may systematically over/under-rate
4. **Corpus coverage gaps** — Our 5K thread subsample doesn't cover rare issues
5. **No production distribution** — Sampled from historical data, not live traffic
6. **First-message only** — We evaluate first-turn quality, not multi-turn resolution

---

## 🔍 Failure Analysis (Top 5)

1. **Multi-issue messages** — Classifier picks one intent, reply only addresses half the problem
2. **Sarcasm** — Literal classification misses implied frustration
3. **Context-dependent follow-ups** — "Still waiting" is unclassifiable without thread history
4. **Policy-specific queries** — LLM may hallucinate Apple policies
5. **Non-English messages** — Replies default to English regardless of input language

---

## 📝 Decision Log

| # | Decision | Why |
|---|----------|-----|
| 1 | AppleSupport as brand | Highest volume (106K tweets), most diverse issues |
| 2 | OpenRouter for LLM API | Model flexibility, single API key |
| 3 | Gemini Flash for pipeline, Claude Sonnet for judge | Avoids self-evaluation bias |
| 4 | 8-12 intents (not 3, not 77) | Balances granularity with evaluation tractability |
| 5 | FAISS over BM25 for retrieval | Semantic similarity outperforms keyword matching for noisy tweets |
| 6 | Conservative escalation default | The cost of a false escalation to a human is lower than the brand damage of an AI hallucinating a bad response to an angry customer. |
| 7 | Stratified + adversarial golden set | Random sampling would miss edge cases |
| 8 | Judge calibration with Cohen's κ | Proves (or disproves) judge reliability |
| 9 | Thread reconstruction from flat tweets | Gives context for how Apple resolves issues |
| 10 | 5K thread subsample | Keeps runtime < 15 min with good variety |
| 11 | Banking77 as structural reference only | Used for taxonomy design principles, not training data |
| 12 | Rule + LLM hybrid escalation | Rules for obvious triggers, LLM for nuance |
| 13 | Replies grounded in historical responses | Prevents unconstrained hallucination |
| 14 | Two baselines (trivial + simple) | Shows LLM value-add over classical ML |
| 15 | Caching/skip logic in pipeline | Reproducibility without re-spending API credits |

---

## 📁 Project Structure

```
Hiver/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
├── run_pipeline.py                    # Single-command entry point
├── data/
│   ├── preprocess.py                  # Data filtering & thread reconstruction
│   └── processed/                     # Generated: cleaned threads, embeddings
├── src/
│   ├── agent.py                       # Core agent (classify, reply, escalate)
│   ├── retrieval.py                   # FAISS-based similar thread retrieval
│   ├── intent_discovery.py            # Clustering + taxonomy generation
│   ├── intent_taxonomy.json           # Generated: finalized intents
│   └── llm_client.py                  # OpenRouter API wrapper
├── evaluation/
│   ├── golden_set.jsonl               # 200 hand-labelled examples
│   ├── sampling_methodology.md        # How we built the golden set
│   ├── build_golden_set.py            # Golden set construction
│   ├── eval_harness.py                # Automated metrics & comparison
│   ├── baselines.py                   # Trivial + simple baselines
│   └── llm_judge.py                   # LLM-as-judge with calibration
├── report/
│   └── REPORT.md                      # Detailed 6-page report
└── results/                           # Generated: metrics, plots, reports
```

---

## 🔧 What I'd Do Next With One More Week

1. **Multi-turn conversation modeling** — Use full thread context, not just first messages
2. **Fine-tune a smaller model** — Distill LLM outputs into a cheaper, faster model
3. **External labeler validation** — Inter-annotator agreement with independent labelers
4. **Structured Apple knowledge base** — Ground replies in actual policies, not just historical tweets
5. **A/B testing framework** — Systematic comparison of prompt strategies and configurations

---

## References

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle)
- **Banking77**: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (HuggingFace)
- **Embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **LLM API**: [OpenRouter](https://openrouter.ai/)
- **AI coding assistant**: Used for code scaffolding. All code reviewed and understood by author.