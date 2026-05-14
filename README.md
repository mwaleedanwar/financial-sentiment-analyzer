# Financial Sentiment Analyzer

Fine-tuned **DistilBERT** for **three-class financial sentiment** (Negative, Positive, Neutral), with a **FastAPI** inference service and a **Next.js** demo UI.

---

## 1. Dataset

**Name:** [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

**Hugging Face identifier:** `zeroshot/twitter-financial-news-sentiment`  
**Direct link:** https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

**Why it is included:** The corpus is **finance-focused** (Twitter financial news style), **English**, and **manually annotated** for **three sentiment classes**, which matches the project goal of a compact, realistic sentiment model for short financial text. It is **available via the Hugging Face `datasets` API**, so training stays reproducible and easy to script. Labels in the card are often described as bearish / bullish / neutral; this project maps them to **Negative (0), Positive (1), Neutral (2)** for a consistent API and UI.

---

## 2. Feature extraction method

**Method:** **Contextual subword representations** from a pretrained **DistilBERT** checkpoint (`distilbert-base-uncased`), produced by the model’s **tokenizer** (WordPiece) and **Transformer encoder**, not hand-built lexicons or bag-of-words features.

**Why this choice:** DistilBERT retains much of BERT’s expressive power with **fewer parameters and faster training/inference**, which suits coursework timelines and local hardware. **Fine-tuning** the pretrained encoder adapts general language representations to **domain-specific financial phrasing** better than training a small model from scratch on this dataset size.

---

## 3. Evaluation measures

During training, the Hugging Face **Trainer** reports:

- **F1 (macro)** — average F1 across classes with equal weight; highlights performance on **rarer classes** and is a standard choice for **multiclass** problems.
- **F1 (weighted)** — F1 averaged by **class support**; aligns more closely with **overall accuracy** when classes are imbalanced.

**Why these:** Together they give a **balanced view** of quality (macro vs. weighted) plus an **intuitive headline number** (accuracy). The training script uses an **80/20 stratified train–test split** after merging available splits so class proportions stay stable in both partitions.

---

## 4. Computational resources

Training was run locally.

---

## Repository layout (high level)

| Path        | Role                                                  |
| ----------- | ----------------------------------------------------- |
| `train.py`  | Fine-tune and create the actual model                 |
| `backend/`  | FastAPI app to use the trained model                  |
| `frontend/` | Next.js App Router UI and optional Vercel proxy route |

---

## Live demo

**Deployment:** https://financial-sentiment-analyzer-pi.vercel.app
