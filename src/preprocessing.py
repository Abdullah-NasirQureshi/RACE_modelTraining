"""RACE preprocessing utilities.

Responsibilities
----------------
1. Load RACE CSV splits with the canonical schema:
     id, article, question, A, B, C, D, answer
2. Lowercase + light punctuation cleanup, tokenisation helpers.
3. Expand each multiple-choice row into 4 supervised verifier examples
   `(article, question, option_text, label in {0, 1})`.
4. Sentence splitting that does not require any external NLP library
   (works with pure regex; safe on Windows + Kaggle + Colab).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "article",
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
)

_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: object) -> str:
    """Lowercase + collapse whitespace + drop punctuation (keeps apostrophes)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: object) -> List[str]:
    """Lowercase, alpha-only word tokens. Cheap and dependency-free."""
    if text is None:
        return []
    return _WORD_RE.findall(str(text).lower())


def split_sentences(article: object) -> List[str]:
    """Best-effort sentence splitter. Returns the original (cased) sentences."""
    if article is None:
        return []
    text = str(article).strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def load_race_csv(
    path: str,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a RACE CSV split, validate the schema, optionally subsample."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RACE CSV not found at {path!r}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"RACE CSV at {path!r} is missing required columns: {missing}. "
            f"Expected: {list(REQUIRED_COLUMNS)}"
        )

    df = df.dropna(subset=["article", "question", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df = df[df["answer"].isin(OPTION_LETTERS)].reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Verifier example expansion
# ---------------------------------------------------------------------------

@dataclass
class VerifierExample:
    """One (article, question, option, label) supervised example."""
    article: str
    question: str
    option_text: str
    option_letter: str
    label: int   # 1 if option_letter == answer, else 0


def expand_to_verifier_examples(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert N RACE rows into 4*N verifier rows.

    Returns
    -------
    X_df : pandas DataFrame with columns
        [article, question, option_text, option_letter, source_idx]
    y    : np.ndarray of {0, 1} labels
    """
    rows: List[dict] = []
    labels: List[int] = []
    for src_idx, r in df.reset_index(drop=True).iterrows():
        gold = str(r["answer"]).strip().upper()
        for letter in OPTION_LETTERS:
            opt_text = r.get(letter)
            if opt_text is None or (isinstance(opt_text, float) and np.isnan(opt_text)):
                continue
            rows.append(
                {
                    "article": r["article"],
                    "question": r["question"],
                    "option_text": str(opt_text),
                    "option_letter": letter,
                    "source_idx": int(src_idx),
                }
            )
            labels.append(1 if letter == gold else 0)

    return pd.DataFrame(rows), np.asarray(labels, dtype=np.int64)


def combined_verifier_text(article: object, question: object, option_text: object) -> str:
    """Concatenation used as the input to the One-Hot vectorizer.

    Article is repeated so its tokens carry more weight than the question /
    option tokens (per the TF-IDF manual section 6.2 advice).
    """
    a = clean_text(article)
    q = clean_text(question)
    o = clean_text(option_text)
    return f"{a} {a} {q} {o}".strip()


# ---------------------------------------------------------------------------
# Wh question-type labelling (used by Naive Bayes, spec 4.2.1)
# ---------------------------------------------------------------------------

WH_LABELS: Tuple[str, ...] = ("what", "who", "where", "when", "why", "how", "other")


def label_question_type(question: object) -> str:
    """Coarse Wh-bucket of the question, used as a NB target."""
    if question is None:
        return "other"
    q = clean_text(question)
    first = q.split(" ", 1)[0] if q else ""
    if first in WH_LABELS:
        return first
    for w in WH_LABELS:
        if f" {w} " in f" {q} ":
            return w
    return "other"


def attach_question_type(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a `q_type` column."""
    df = df.copy()
    df["q_type"] = df["question"].map(label_question_type)
    return df


# ---------------------------------------------------------------------------
# Convenience iterators
# ---------------------------------------------------------------------------

def iter_options(row: pd.Series) -> Iterable[Tuple[str, str]]:
    """Yield (letter, option_text) for a RACE row."""
    for letter in OPTION_LETTERS:
        yield letter, str(row.get(letter, ""))
