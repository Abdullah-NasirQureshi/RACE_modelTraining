"""Model B training: distractor ranker + graduated hint ranker.

Pipeline (per spec section 5)
-----------------------------
1. Distractor candidate extraction from the passage (regex-based n-grams +
   frequency cutoff, no external NLP dependency).
2. Logistic Regression ranker over per-candidate features:
     - One-Hot cosine similarity to the gold answer
     - Character-level match score
     - Passage frequency
     - Length ratio
     - Stopword ratio
   Positive class = real RACE distractors; Negative class = random non-answer
   passage candidates not present in the option set.
3. Diversity-aware top-3 selection at inference time.
4. Hint ranker (Logistic Regression) over per-sentence features:
     - Cosine similarity to question
     - Overlap with the answer tokens
     - Position in passage
     - Sentence length
   Trained as a regression-style proxy: sentences containing the gold answer
   tokens get higher relevance.

Outputs (under models/model_b/traditional/):
    distractor_lr.pkl
    distractor_features_meta.json
    hint_lr.pkl
    metrics_b.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from .features import build_onehot_vectorizer
from .preprocessing import (
    OPTION_LETTERS,
    clean_text,
    split_sentences,
    tokenize,
)


MODEL_B_DIR = os.path.join("models", "model_b", "traditional")

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "you", "he", "she", "we", "us", "our", "your", "my", "me", "his",
    "her", "him", "have", "has", "had", "do", "does", "did", "not", "no",
    "so", "if", "then", "than", "into", "out", "up", "down", "over", "under",
    "very", "more", "most", "some", "any", "all",
}

_NGRAM_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidate_phrases(
    article: str,
    min_words: int = 1,
    max_words: int = 4,
    min_token_len: int = 2,
    max_candidates: int = 80,
) -> List[str]:
    """Extract candidate phrases (1..4-grams) from the passage.

    Strategy (spec 5.3.1 step 1, "no NLP tools required"):
      - tokenise
      - generate n-grams of length 1..max_words
      - drop n-grams that are entirely stopwords
      - cap total candidates by passage frequency
    """
    if not article:
        return []
    tokens = _NGRAM_RE.findall(str(article))
    if not tokens:
        return []
    tokens_lower = [t.lower() for t in tokens]

    # Frequency of each lowercase token (for ranking)
    counts: Dict[str, int] = {}
    for t in tokens_lower:
        counts[t] = counts.get(t, 0) + 1

    seen: Dict[str, int] = {}   # candidate phrase -> passage count
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i : i + n]
            ngram_lower = [t.lower() for t in ngram_tokens]
            if any(len(t) < min_token_len for t in ngram_lower):
                continue
            if all(t in _STOPWORDS for t in ngram_lower):
                continue
            phrase = " ".join(ngram_tokens)
            seen[phrase] = seen.get(phrase, 0) + 1

    # Sort by frequency desc, then phrase length asc
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [phrase for phrase, _ in ranked[:max_candidates]]


# ---------------------------------------------------------------------------
# Distractor candidate features
# ---------------------------------------------------------------------------

@dataclass
class DistractorFeatureRow:
    cosine_to_answer: float
    char_match: float
    passage_freq: float
    length_ratio: float
    stopword_ratio: float
    overlap_with_answer: float

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.cosine_to_answer,
                self.char_match,
                self.passage_freq,
                self.length_ratio,
                self.stopword_ratio,
                self.overlap_with_answer,
            ],
            dtype=np.float32,
        )


DISTRACTOR_FEATURE_NAMES = [
    "cosine_to_answer",
    "char_match",
    "passage_freq",
    "length_ratio",
    "stopword_ratio",
    "overlap_with_answer",
]


def _char_ngram_match(a: str, b: str, n: int = 3) -> float:
    a_low, b_low = a.lower().strip(), b.lower().strip()
    if not a_low or not b_low:
        return 0.0
    a_set = {a_low[i : i + n] for i in range(max(0, len(a_low) - n + 1))} or {a_low}
    b_set = {b_low[i : i + n] for i in range(max(0, len(b_low) - n + 1))} or {b_low}
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def _passage_frequency(phrase: str, article: str) -> int:
    if not phrase or not article:
        return 0
    return article.lower().count(phrase.lower())


def _stopword_ratio(phrase: str) -> float:
    toks = phrase.lower().split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _STOPWORDS) / len(toks)


def _overlap(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_distractor_feature_row(
    candidate: str,
    answer: str,
    article: str,
    onehot_vectorizer,
) -> DistractorFeatureRow:
    cand_clean = clean_text(candidate)
    ans_clean = clean_text(answer)
    if cand_clean and ans_clean:
        try:
            X = onehot_vectorizer.transform([cand_clean, ans_clean])
            cos = float(cosine_similarity(X[0], X[1])[0, 0])
        except Exception:
            cos = 0.0
    else:
        cos = 0.0
    char_m = _char_ngram_match(candidate, answer)
    freq = _passage_frequency(candidate, article)
    a_toks = tokenize(answer)
    c_toks = tokenize(candidate)
    overlap = _overlap(c_toks, a_toks)
    len_ratio = len(c_toks) / max(1, len(a_toks))
    sw = _stopword_ratio(candidate)
    return DistractorFeatureRow(
        cosine_to_answer=cos,
        char_match=char_m,
        passage_freq=float(freq),
        length_ratio=float(len_ratio),
        stopword_ratio=float(sw),
        overlap_with_answer=float(overlap),
    )


# ---------------------------------------------------------------------------
# Build the (X, y) training set for the distractor ranker
# ---------------------------------------------------------------------------

def build_distractor_training_data(
    df: pd.DataFrame,
    onehot_vectorizer,
    n_negatives_per_positive: int = 3,
    seed: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    labels: List[int] = []

    iter_df = df if max_rows is None else df.head(max_rows)
    for _, r in iter_df.iterrows():
        article = str(r["article"])
        gold_letter = str(r["answer"]).strip().upper()
        gold_text = str(r.get(gold_letter, ""))

        # Positive examples: the *real* distractors (the 3 non-gold options)
        positives: List[str] = []
        for letter in OPTION_LETTERS:
            if letter == gold_letter:
                continue
            text = str(r.get(letter, "")).strip()
            if text:
                positives.append(text)

        for pos in positives:
            feats = compute_distractor_feature_row(pos, gold_text, article, onehot_vectorizer)
            rows.append(feats.to_vector())
            labels.append(1)

        # Negative examples: random passage candidates that are NOT
        # any of the four MCQ options (so they're plausible-but-not-gold).
        candidates = extract_candidate_phrases(article)
        option_texts_lower = {
            str(r.get(letter, "")).strip().lower() for letter in OPTION_LETTERS
        }
        non_options = [c for c in candidates if c.lower() not in option_texts_lower]
        if non_options:
            n_neg = min(n_negatives_per_positive * max(1, len(positives)), len(non_options))
            chosen = rng.choice(len(non_options), size=n_neg, replace=False)
            for idx in chosen:
                feats = compute_distractor_feature_row(
                    non_options[idx], gold_text, article, onehot_vectorizer
                )
                rows.append(feats.to_vector())
                labels.append(0)

    if not rows:
        return np.zeros((0, len(DISTRACTOR_FEATURE_NAMES)), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Hint ranker (per-sentence features, regression-style logistic)
# ---------------------------------------------------------------------------

@dataclass
class HintFeatureRow:
    cosine_to_question: float
    overlap_with_answer: float
    position_norm: float
    length_norm: float

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.cosine_to_question,
                self.overlap_with_answer,
                self.position_norm,
                self.length_norm,
            ],
            dtype=np.float32,
        )


HINT_FEATURE_NAMES = [
    "cosine_to_question",
    "overlap_with_answer",
    "position_norm",
    "length_norm",
]


def compute_hint_feature_row(
    sentence: str,
    question: str,
    answer: str,
    position: int,
    n_sentences: int,
    onehot_vectorizer,
) -> HintFeatureRow:
    s_clean = clean_text(sentence)
    q_clean = clean_text(question)
    if s_clean and q_clean:
        try:
            X = onehot_vectorizer.transform([s_clean, q_clean])
            cos = float(cosine_similarity(X[0], X[1])[0, 0])
        except Exception:
            cos = 0.0
    else:
        cos = 0.0
    overlap = _overlap(tokenize(sentence), tokenize(answer))
    length = len(tokenize(sentence))
    return HintFeatureRow(
        cosine_to_question=cos,
        overlap_with_answer=float(overlap),
        position_norm=float(position) / float(max(1, n_sentences - 1)) if n_sentences > 1 else 0.0,
        length_norm=float(length) / 50.0,  # divide by typical sentence length
    )


def build_hint_training_data(
    df: pd.DataFrame,
    onehot_vectorizer,
    seed: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hint relevance label: 1 if sentence shares >=1 content token with the
    gold answer, else 0. This is the standard 'extractive evidence' proxy."""
    rows: List[np.ndarray] = []
    labels: List[int] = []
    iter_df = df if max_rows is None else df.head(max_rows)
    for _, r in iter_df.iterrows():
        article = str(r["article"])
        question = str(r["question"])
        gold_letter = str(r["answer"]).strip().upper()
        gold_text = str(r.get(gold_letter, ""))
        sentences = split_sentences(article)
        if not sentences:
            continue
        gold_tokens = set(tokenize(gold_text)) - _STOPWORDS

        for pos, sent in enumerate(sentences):
            feats = compute_hint_feature_row(
                sent, question, gold_text, pos, len(sentences), onehot_vectorizer
            )
            label = 1 if (set(tokenize(sent)) & gold_tokens) else 0
            rows.append(feats.to_vector())
            labels.append(label)

    if not rows:
        return np.zeros((0, len(HINT_FEATURE_NAMES)), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelBMetrics:
    distractor_precision: float
    distractor_recall: float
    distractor_f1: float
    distractor_top1_acc: float
    hint_precision: float
    hint_r2: float
    n_distractor_train: int
    n_hint_train: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_model_b(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42,
    output_dir: str = MODEL_B_DIR,
    max_rows: Optional[int] = None,
    onehot_vectorizer=None,
) -> ModelBMetrics:
    _ensure_dir(output_dir)

    # Re-use a vectorizer if Model A already built one; otherwise fit a small
    # one on the article corpus.
    if onehot_vectorizer is None:
        print("[Model B] fitting auxiliary One-Hot vectorizer ...")
        onehot_vectorizer = build_onehot_vectorizer(
            train_df["article"].astype(str).tolist(),
            max_features=10_000,
        )

    # ---- Distractor ranker -------------------------------------------------
    print("[Model B] building distractor training data ...")
    X_dist_tr, y_dist_tr = build_distractor_training_data(
        train_df, onehot_vectorizer, seed=seed, max_rows=max_rows
    )
    X_dist_val, y_dist_val = build_distractor_training_data(
        val_df, onehot_vectorizer, seed=seed, max_rows=max_rows
    )
    print(
        f"[Model B] distractor train rows: {len(X_dist_tr)}  val rows: {len(X_dist_val)}"
    )

    if len(np.unique(y_dist_tr)) < 2:
        # Degenerate; ensure model is callable
        print("[Model B] WARNING: not enough class variety in distractor labels; using fallback.")
        X_fallback = np.zeros((4, len(DISTRACTOR_FEATURE_NAMES)), dtype=np.float32)
        y_fallback = np.array([0, 1, 0, 1], dtype=np.int64)
        dist_lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=seed)
        dist_lr.fit(X_fallback, y_fallback)
        d_prec = d_rec = d_f1 = d_top1 = 0.0
    else:
        dist_lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=300, random_state=seed
        )
        dist_lr.fit(X_dist_tr, y_dist_tr)
        if len(X_dist_val):
            d_pred = dist_lr.predict(X_dist_val)
            d_prec = float(precision_score(y_dist_val, d_pred, zero_division=0))
            d_rec = float(recall_score(y_dist_val, d_pred, zero_division=0))
            d_f1 = float(f1_score(y_dist_val, d_pred, zero_division=0))
        else:
            d_prec = d_rec = d_f1 = 0.0

        # Top-1 distractor accuracy: fraction of dev samples where the
        # top-ranked candidate is NOT the gold answer.
        d_top1_correct = 0
        d_top1_total = 0
        for _, r in val_df.iterrows():
            article = str(r["article"])
            gold_letter = str(r["answer"]).strip().upper()
            gold_text = str(r.get(gold_letter, "")).strip().lower()
            cands = extract_candidate_phrases(article)
            if not cands or not gold_text:
                continue
            feats = np.vstack(
                [
                    compute_distractor_feature_row(c, gold_text, article, onehot_vectorizer).to_vector()
                    for c in cands
                ]
            )
            try:
                proba = dist_lr.predict_proba(feats)[:, 1]
            except Exception:
                proba = dist_lr.predict(feats).astype(float)
            top_idx = int(np.argmax(proba))
            top_text = cands[top_idx].lower()
            d_top1_total += 1
            if top_text != gold_text:
                d_top1_correct += 1
        d_top1 = d_top1_correct / d_top1_total if d_top1_total else 0.0

    print(
        f"[Model B] distractor: prec={d_prec:.3f}  recall={d_rec:.3f}  "
        f"f1={d_f1:.3f}  top1!=gold={d_top1:.3f}"
    )

    # ---- Hint ranker -------------------------------------------------------
    print("[Model B] building hint training data ...")
    X_hint_tr, y_hint_tr = build_hint_training_data(
        train_df, onehot_vectorizer, seed=seed, max_rows=max_rows
    )
    X_hint_val, y_hint_val = build_hint_training_data(
        val_df, onehot_vectorizer, seed=seed, max_rows=max_rows
    )
    print(
        f"[Model B] hint train rows: {len(X_hint_tr)}  val rows: {len(X_hint_val)}"
    )

    if len(np.unique(y_hint_tr)) < 2:
        print("[Model B] WARNING: not enough hint label variety; using fallback.")
        X_fallback = np.zeros((4, len(HINT_FEATURE_NAMES)), dtype=np.float32)
        y_fallback = np.array([0, 1, 0, 1], dtype=np.int64)
        hint_lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=seed)
        hint_lr.fit(X_fallback, y_fallback)
        h_prec = h_r2 = 0.0
    else:
        hint_lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=300, random_state=seed
        )
        hint_lr.fit(X_hint_tr, y_hint_tr)
        if len(X_hint_val):
            h_pred = hint_lr.predict(X_hint_val)
            try:
                h_proba = hint_lr.predict_proba(X_hint_val)[:, 1]
            except Exception:
                h_proba = h_pred.astype(float)
            h_prec = float(precision_score(y_hint_val, h_pred, zero_division=0))
            try:
                h_r2 = float(r2_score(y_hint_val, h_proba))
            except Exception:
                h_r2 = 0.0
        else:
            h_prec = h_r2 = 0.0

    print(f"[Model B] hint: precision={h_prec:.3f}  R2={h_r2:.3f}")

    # ---- Persist artifacts -------------------------------------------------
    _ensure_dir(output_dir)
    print(f"[Model B] saving artifacts to {output_dir} ...")
    joblib.dump(dist_lr, os.path.join(output_dir, "distractor_lr.pkl"))
    joblib.dump(hint_lr, os.path.join(output_dir, "hint_lr.pkl"))

    # Save the auxiliary vectorizer too (so inference does not need Model A)
    joblib.dump(onehot_vectorizer, os.path.join(output_dir, "onehot_vectorizer.pkl"))

    feature_meta = {
        "distractor_features": DISTRACTOR_FEATURE_NAMES,
        "hint_features": HINT_FEATURE_NAMES,
    }
    with open(os.path.join(output_dir, "distractor_features_meta.json"), "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)

    metrics = ModelBMetrics(
        distractor_precision=d_prec,
        distractor_recall=d_rec,
        distractor_f1=d_f1,
        distractor_top1_acc=d_top1,
        hint_precision=h_prec,
        hint_r2=h_r2,
        n_distractor_train=int(len(X_dist_tr)),
        n_hint_train=int(len(X_hint_tr)),
    )
    with open(os.path.join(output_dir, "metrics_b.json"), "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print("[Model B] done.")
    return metrics
