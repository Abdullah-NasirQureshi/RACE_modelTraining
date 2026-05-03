"""Feature engineering for the RACE classical-ML pipeline.

Primary text representation (per the project spec):
    One-Hot Encoding  ==  CountVectorizer(binary=True)

Each document becomes a sparse {0, 1} vector indicating presence/absence of
each vocabulary term. We also support a TF-IDF representation via
`build_tfidf_vectorizer`, used only when the optional ablation flag
`--use_tfidf` is passed to `src.train_all`.

This module also provides handcrafted lexical features that are concatenated
with the sparse text vector, which generally help linear models on RACE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import (
    clean_text,
    combined_verifier_text,
    tokenize,
)


# ---------------------------------------------------------------------------
# Vectorizer factories
# ---------------------------------------------------------------------------

DEFAULT_MAX_FEATURES = 10_000


def build_onehot_vectorizer(
    corpus: Iterable[str],
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> CountVectorizer:
    """Fit a binary CountVectorizer (==> One-Hot Encoding of tokens)."""
    vec = CountVectorizer(
        binary=True,                  # <-- One-Hot semantics
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
    )
    vec.fit(list(corpus))
    return vec


def build_tfidf_vectorizer(
    corpus: Iterable[str],
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """Optional ablation feature: scikit-learn TfidfVectorizer."""
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
    )
    vec.fit(list(corpus))
    return vec


# ---------------------------------------------------------------------------
# Handcrafted lexical features for the verifier
# ---------------------------------------------------------------------------

@dataclass
class HandcraftedFeatures:
    """Handcrafted scalar features for one (article, question, option) tuple."""
    article_len: float
    question_len: float
    option_len: float
    overlap_q_option: float       # Jaccard overlap of question and option tokens
    overlap_a_option: float       # Jaccard overlap of article and option tokens
    cosine_q_option: float        # cosine similarity in the One-Hot space
    cosine_a_option: float        # cosine similarity of article vs option

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.article_len,
                self.question_len,
                self.option_len,
                self.overlap_q_option,
                self.overlap_a_option,
                self.cosine_q_option,
                self.cosine_a_option,
            ],
            dtype=np.float32,
        )


HANDCRAFTED_FEATURE_DIM = 7


def _safe_jaccard(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _cosine_via_vectorizer(text_a: str, text_b: str, vectorizer) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    X = vectorizer.transform([text_a, text_b])
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)


def compute_handcrafted_features(
    article: str,
    question: str,
    option_text: str,
    vectorizer=None,
) -> HandcraftedFeatures:
    """Compute the 7 handcrafted scalar features for one example."""
    a_tokens = tokenize(article)
    q_tokens = tokenize(question)
    o_tokens = tokenize(option_text)

    cosine_q_option = 0.0
    cosine_a_option = 0.0
    if vectorizer is not None:
        try:
            cosine_q_option = _cosine_via_vectorizer(
                clean_text(question), clean_text(option_text), vectorizer
            )
            cosine_a_option = _cosine_via_vectorizer(
                clean_text(article), clean_text(option_text), vectorizer
            )
        except Exception:
            cosine_q_option = 0.0
            cosine_a_option = 0.0

    return HandcraftedFeatures(
        article_len=float(len(a_tokens)),
        question_len=float(len(q_tokens)),
        option_len=float(len(o_tokens)),
        overlap_q_option=_safe_jaccard(q_tokens, o_tokens),
        overlap_a_option=_safe_jaccard(a_tokens, o_tokens),
        cosine_q_option=cosine_q_option,
        cosine_a_option=cosine_a_option,
    )


def batch_handcrafted(
    articles: Sequence[str],
    questions: Sequence[str],
    options: Sequence[str],
    vectorizer=None,
) -> np.ndarray:
    """Vectorised handcrafted feature computation. Returns shape (N, 7)."""
    n = len(articles)
    out = np.zeros((n, HANDCRAFTED_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        feats = compute_handcrafted_features(
            articles[i], questions[i], options[i], vectorizer=vectorizer
        )
        out[i] = feats.to_vector()
    return out


# ---------------------------------------------------------------------------
# Stacking the sparse text matrix with the dense handcrafted matrix
# ---------------------------------------------------------------------------

def stack_text_and_handcrafted(
    text_matrix: sp.spmatrix,
    handcrafted_matrix: np.ndarray,
) -> sp.csr_matrix:
    """Horizontally stack [sparse_text | sparse_handcrafted] -> CSR."""
    if text_matrix.shape[0] != handcrafted_matrix.shape[0]:
        raise ValueError(
            "Row count mismatch: text "
            f"{text_matrix.shape[0]} vs handcrafted {handcrafted_matrix.shape[0]}"
        )
    hc_sparse = sp.csr_matrix(handcrafted_matrix.astype(np.float32))
    return sp.hstack([text_matrix, hc_sparse]).tocsr()


# ---------------------------------------------------------------------------
# Convenience: build the verifier feature matrix from a verifier DataFrame
# ---------------------------------------------------------------------------

def build_verifier_features(
    df,
    vectorizer,
) -> sp.csr_matrix:
    """From a verifier DataFrame (article, question, option_text), build X."""
    texts = [
        combined_verifier_text(a, q, o)
        for a, q, o in zip(df["article"], df["question"], df["option_text"])
    ]
    X_text = vectorizer.transform(texts)
    X_hc = batch_handcrafted(
        df["article"].tolist(),
        df["question"].tolist(),
        df["option_text"].tolist(),
        vectorizer=vectorizer,
    )
    return stack_text_and_handcrafted(X_text, X_hc)
