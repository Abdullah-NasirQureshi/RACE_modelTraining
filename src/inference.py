"""Unified runtime API used by the Streamlit UI.

Loads all trained artifacts (lazily) and exposes a small, stable surface:

    artifacts = load_artifacts()              # caches loaded models
    proba   = verify_answer(article, q, opt, artifacts)
    letter  = predict_best_option(article, q, options_dict, artifacts)
    qs      = generate_questions(article, answer, artifacts)
    distrs  = generate_distractors(article, answer, artifacts)
    hints   = rank_hints(article, question, answer, artifacts)
    qtype   = predict_question_type(question, artifacts)

Every function gracefully degrades when artifacts are missing so the UI is
still usable on demo_data (used as a fallback by the app).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .features import (
    HANDCRAFTED_FEATURE_DIM,
    batch_handcrafted,
    stack_text_and_handcrafted,
)
from .model_b_train import (
    compute_distractor_feature_row,
    compute_hint_feature_row,
    extract_candidate_phrases,
)
from .preprocessing import (
    OPTION_LETTERS,
    clean_text,
    combined_verifier_text,
    label_question_type,
    split_sentences,
    tokenize,
)
from .question_generation import (
    GeneratedQuestion,
    generate_and_rank,
)


MODEL_A_DIR = os.path.join("models", "model_a", "traditional")
MODEL_B_DIR = os.path.join("models", "model_b", "traditional")


# ---------------------------------------------------------------------------
# Artifact container
# ---------------------------------------------------------------------------

@dataclass
class Artifacts:
    """Bundle of all trained artifacts. Any of these may be None if missing."""
    vectorizer: object = None              # CountVectorizer(binary=True) or TfidfVectorizer
    lr_verifier: object = None
    svm_verifier: object = None
    ensemble: object = None
    nb_qtype: object = None
    nb_qtype_vec: object = None
    qgen_ranker: object = None

    distractor_lr: object = None
    hint_lr: object = None
    aux_onehot_vectorizer: object = None   # used by Model B at inference time

    available: Dict[str, bool] = field(default_factory=dict)

    @property
    def is_trained(self) -> bool:
        return self.vectorizer is not None and (
            self.lr_verifier is not None or self.ensemble is not None
        )


def _try_load(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:                  # pragma: no cover
            print(f"[inference] failed to load {path}: {e}")
            return None
    return None


_CACHED_ARTIFACTS: Optional[Artifacts] = None


def load_artifacts(force_reload: bool = False) -> Artifacts:
    """Load all artifacts with on-disk caching."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None and not force_reload:
        return _CACHED_ARTIFACTS

    a = Artifacts()
    a.vectorizer = _try_load(os.path.join(MODEL_A_DIR, "vectorizer.pkl"))
    a.lr_verifier = _try_load(os.path.join(MODEL_A_DIR, "lr_verifier.pkl"))
    a.svm_verifier = _try_load(os.path.join(MODEL_A_DIR, "svm_verifier.pkl"))
    a.ensemble = _try_load(os.path.join(MODEL_A_DIR, "ensemble.pkl"))
    a.nb_qtype = _try_load(os.path.join(MODEL_A_DIR, "nb_qtype.pkl"))
    a.nb_qtype_vec = _try_load(os.path.join(MODEL_A_DIR, "nb_qtype_vec.pkl"))
    a.qgen_ranker = _try_load(os.path.join(MODEL_A_DIR, "qgen_ranker.pkl"))

    a.distractor_lr = _try_load(os.path.join(MODEL_B_DIR, "distractor_lr.pkl"))
    a.hint_lr = _try_load(os.path.join(MODEL_B_DIR, "hint_lr.pkl"))
    a.aux_onehot_vectorizer = _try_load(os.path.join(MODEL_B_DIR, "onehot_vectorizer.pkl"))

    a.available = {
        "vectorizer": a.vectorizer is not None,
        "lr_verifier": a.lr_verifier is not None,
        "svm_verifier": a.svm_verifier is not None,
        "ensemble": a.ensemble is not None,
        "nb_qtype": a.nb_qtype is not None and a.nb_qtype_vec is not None,
        "qgen_ranker": a.qgen_ranker is not None,
        "distractor_lr": a.distractor_lr is not None,
        "hint_lr": a.hint_lr is not None,
        "aux_onehot_vectorizer": a.aux_onehot_vectorizer is not None,
    }

    _CACHED_ARTIFACTS = a
    return a


def _resolve_onehot_vectorizer(artifacts: Artifacts):
    """Pick whichever vectorizer is available (Model A's or Model B's)."""
    return artifacts.vectorizer or artifacts.aux_onehot_vectorizer


# ---------------------------------------------------------------------------
# Model A inference
# ---------------------------------------------------------------------------

def verify_answer(
    article: str,
    question: str,
    option_text: str,
    artifacts: Optional[Artifacts] = None,
) -> Tuple[float, float]:
    """Return (probability_correct, latency_ms).

    Falls back to a token-overlap heuristic when no model is loaded.
    """
    artifacts = artifacts or load_artifacts()
    t0 = time.time()

    if artifacts.is_trained and artifacts.vectorizer is not None:
        text = combined_verifier_text(article, question, option_text)
        X_text = artifacts.vectorizer.transform([text])

        # Try ensemble first (text-only space), then LR (with handcrafted)
        if artifacts.ensemble is not None:
            try:
                p = float(artifacts.ensemble.predict_proba(X_text)[0, 1])
                return p, (time.time() - t0) * 1000.0
            except Exception:
                pass

        if artifacts.lr_verifier is not None:
            X_hc = batch_handcrafted(
                [article], [question], [option_text], vectorizer=artifacts.vectorizer
            )
            X_full = stack_text_and_handcrafted(X_text, X_hc)
            try:
                p = float(artifacts.lr_verifier.predict_proba(X_full)[0, 1])
                return p, (time.time() - t0) * 1000.0
            except Exception:
                pass

    # Heuristic fallback ------------------------------------------------------
    a_tokens = set(tokenize(article))
    q_tokens = set(tokenize(question))
    o_tokens = set(tokenize(option_text))
    if not o_tokens:
        return 0.0, (time.time() - t0) * 1000.0
    overlap_a = len(a_tokens & o_tokens) / max(1, len(o_tokens))
    overlap_q = len(q_tokens & o_tokens) / max(1, len(o_tokens))
    p = max(0.0, min(1.0, 0.6 * overlap_a + 0.4 * overlap_q))
    return p, (time.time() - t0) * 1000.0


def predict_best_option(
    article: str,
    question: str,
    options: Dict[str, str],
    artifacts: Optional[Artifacts] = None,
) -> Tuple[str, Dict[str, float], float]:
    """Return (best_letter, per_option_probability, latency_ms)."""
    artifacts = artifacts or load_artifacts()
    t0 = time.time()
    scores: Dict[str, float] = {}
    for letter, text in options.items():
        p, _ = verify_answer(article, question, text, artifacts)
        scores[letter] = p
    best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "A"
    return best, scores, (time.time() - t0) * 1000.0


def predict_question_type(
    question: str,
    artifacts: Optional[Artifacts] = None,
) -> str:
    artifacts = artifacts or load_artifacts()
    if artifacts.nb_qtype is not None and artifacts.nb_qtype_vec is not None:
        try:
            X = artifacts.nb_qtype_vec.transform([str(question)])
            return str(artifacts.nb_qtype.predict(X)[0])
        except Exception:
            pass
    return label_question_type(question)


def generate_questions(
    article: str,
    answer: str,
    artifacts: Optional[Artifacts] = None,
    top_k: int = 3,
) -> List[GeneratedQuestion]:
    artifacts = artifacts or load_artifacts()
    return generate_and_rank(article, answer, ranker=artifacts.qgen_ranker, top_k=top_k)


# ---------------------------------------------------------------------------
# Model B inference
# ---------------------------------------------------------------------------

def _diversity_filter(
    selected: List[str],
    candidate: str,
    threshold: float = 0.5,
) -> bool:
    """Return True if `candidate` is sufficiently different from already-picked."""
    cand_tok = set(tokenize(candidate))
    for s in selected:
        s_tok = set(tokenize(s))
        if not s_tok:
            continue
        jacc = len(cand_tok & s_tok) / max(1, len(cand_tok | s_tok))
        if jacc > threshold:
            return False
    return True


def generate_distractors(
    article: str,
    answer: str,
    artifacts: Optional[Artifacts] = None,
    n_distractors: int = 3,
    diversity_threshold: float = 0.5,
) -> List[str]:
    """Return up to `n_distractors` plausible-but-incorrect distractor phrases."""
    artifacts = artifacts or load_artifacts()
    candidates = extract_candidate_phrases(article)
    if not candidates:
        return []

    # Drop any candidate that exactly matches the answer.
    answer_l = clean_text(answer)
    candidates = [c for c in candidates if clean_text(c) != answer_l]
    if not candidates:
        return []

    onehot = _resolve_onehot_vectorizer(artifacts)
    if artifacts.distractor_lr is not None and onehot is not None:
        feats = np.vstack(
            [
                compute_distractor_feature_row(c, answer, article, onehot).to_vector()
                for c in candidates
            ]
        )
        try:
            scores = artifacts.distractor_lr.predict_proba(feats)[:, 1]
        except Exception:
            scores = artifacts.distractor_lr.predict(feats).astype(float)
    else:
        # Fallback: rank by frequency * inverse-overlap with answer
        ans_toks = set(tokenize(answer))
        scores = np.asarray(
            [
                article.lower().count(c.lower())
                * (1.0 - len(set(tokenize(c)) & ans_toks) / max(1, len(set(tokenize(c)) | ans_toks)))
                for c in candidates
            ],
            dtype=np.float32,
        )

    order = np.argsort(-scores)
    picked: List[str] = []
    for idx in order:
        cand = candidates[int(idx)]
        if not _diversity_filter(picked, cand, threshold=diversity_threshold):
            continue
        picked.append(cand)
        if len(picked) >= n_distractors:
            break

    while len(picked) < n_distractors and len(picked) < len(candidates):
        for c in candidates:
            if c not in picked and clean_text(c) != answer_l:
                picked.append(c)
                if len(picked) >= n_distractors:
                    break
    return picked[:n_distractors]


def rank_hints(
    article: str,
    question: str,
    answer: str,
    artifacts: Optional[Artifacts] = None,
    n_hints: int = 3,
) -> List[str]:
    """Return graduated hints from most general (Hint 1) to most specific (Hint 3).

    The ranker scores each sentence by relevance; we then pick:
      Hint 1: the LOWEST-scoring informative sentence (broad context)
      Hint 2: a MID-scoring sentence (more specific clue)
      Hint 3: the HIGHEST-scoring sentence (near-explicit evidence)
    """
    artifacts = artifacts or load_artifacts()
    sentences = split_sentences(article)
    if not sentences:
        return []

    onehot = _resolve_onehot_vectorizer(artifacts)
    n = len(sentences)
    if artifacts.hint_lr is not None and onehot is not None:
        feats = np.vstack(
            [
                compute_hint_feature_row(s, question, answer, i, n, onehot).to_vector()
                for i, s in enumerate(sentences)
            ]
        )
        try:
            scores = artifacts.hint_lr.predict_proba(feats)[:, 1]
        except Exception:
            scores = artifacts.hint_lr.predict(feats).astype(float)
    else:
        # Fallback: cosine-similarity-by-overlap with the question and answer
        scores = np.asarray(
            [
                0.5 * len(set(tokenize(s)) & set(tokenize(question))) / max(1, len(set(tokenize(s)) | set(tokenize(question))))
                + 0.5 * len(set(tokenize(s)) & set(tokenize(answer))) / max(1, len(set(tokenize(s)) | set(tokenize(answer))))
                for s in sentences
            ],
            dtype=np.float32,
        )

    if n_hints <= 0:
        return []

    sorted_idx = np.argsort(scores)  # ascending
    if len(sorted_idx) == 1:
        return [sentences[int(sorted_idx[0])]]

    # Hint 1 -> lowest-scoring (most general clue)
    # Hint 3 -> highest-scoring (near-explicit)
    hints: List[str] = []
    if n_hints >= 1:
        hints.append(sentences[int(sorted_idx[0])])
    if n_hints >= 2:
        mid = int(sorted_idx[len(sorted_idx) // 2])
        if sentences[mid] not in hints:
            hints.append(sentences[mid])
    if n_hints >= 3:
        top = int(sorted_idx[-1])
        if sentences[top] not in hints:
            hints.append(sentences[top])

    # Pad with remaining unique sentences if we need more
    if len(hints) < n_hints:
        for idx in sorted_idx[::-1]:
            s = sentences[int(idx)]
            if s not in hints:
                hints.append(s)
            if len(hints) >= n_hints:
                break

    return hints[:n_hints]
