"""Template-based question generator (Model A, spec 4.2.3).

Strategy
--------
1. Score each sentence in the passage by lexical overlap with the gold
   answer span.
2. Apply Wh-templates to transform the top-ranked sentence into a question.
3. Rank generated candidates with a trained Random Forest scoring fluency
   proxies (length, function-word ratio, overlap).

The ranker is trained on dataset-derived signals: real RACE questions
become positive examples, mechanically-generated noise becomes negative.

This module is fully classical-ML — no neural networks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .preprocessing import (
    clean_text,
    split_sentences,
    tokenize,
)


_FUNCTION_WORDS = {
    "the", "a", "an", "of", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
}


# ---------------------------------------------------------------------------
# Wh-template question generator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedQuestion:
    question_text: str
    template: str
    source_sentence: str
    score: float = 0.0


def _overlap_score(sentence: str, answer: str) -> float:
    s_tok = set(tokenize(sentence))
    a_tok = set(tokenize(answer))
    if not s_tok or not a_tok:
        return 0.0
    return len(s_tok & a_tok) / max(1, len(a_tok))


def _starts_with_capital_pronoun(sentence: str) -> bool:
    first = sentence.strip().split(" ", 1)[0] if sentence.strip() else ""
    return first[:1].isupper()


def _replace_first_named_entity_like(sentence: str) -> Optional[str]:
    """Very simple heuristic: replace the first capitalised non-leading token
    with 'who/what'. Returns None if no candidate found."""
    tokens = sentence.split()
    if not tokens:
        return None
    for i, tok in enumerate(tokens[1:], start=1):
        clean = re.sub(r"[^a-zA-Z]", "", tok)
        if clean and clean[0].isupper():
            tokens[i] = "who"
            return " ".join(tokens)
    return None


_TEMPLATES = [
    ("what", "What is described in the passage about \"{topic}\"?"),
    ("who", "Who or what is associated with \"{topic}\" in the passage?"),
    ("where", "Where does the passage say \"{topic}\" takes place?"),
    ("when", "When does the passage indicate \"{topic}\" happens?"),
    ("why", "Why is \"{topic}\" important according to the passage?"),
    ("how", "How is \"{topic}\" described in the passage?"),
]


def generate_questions(
    article: str,
    answer: str,
    top_k_sentences: int = 3,
) -> List[GeneratedQuestion]:
    """Produce candidate Wh-questions from the passage."""
    sents = split_sentences(article)
    if not sents:
        return []
    scored = sorted(
        sents,
        key=lambda s: _overlap_score(s, answer),
        reverse=True,
    )[:top_k_sentences]

    candidates: List[GeneratedQuestion] = []
    topic_terms = [t for t in tokenize(answer) if t not in _FUNCTION_WORDS][:3]
    topic = " ".join(topic_terms) if topic_terms else (answer[:40].strip() or "the topic")

    for sentence in scored:
        for tag, tmpl in _TEMPLATES:
            q = tmpl.format(topic=topic)
            candidates.append(
                GeneratedQuestion(
                    question_text=q,
                    template=tag,
                    source_sentence=sentence,
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Random Forest ranker for generated questions
# ---------------------------------------------------------------------------

def _fluency_features(question: str, source_sentence: str, answer: str) -> np.ndarray:
    q_tok = tokenize(question)
    s_tok = tokenize(source_sentence)
    a_tok = tokenize(answer)
    n = max(1, len(q_tok))
    fn_ratio = sum(1 for t in q_tok if t in _FUNCTION_WORDS) / n
    overlap_qs = len(set(q_tok) & set(s_tok)) / max(1, len(set(q_tok) | set(s_tok)))
    overlap_qa = len(set(q_tok) & set(a_tok)) / max(1, len(set(q_tok) | set(a_tok)))
    starts_wh = float(q_tok[0] in {"what", "who", "where", "when", "why", "how"} if q_tok else False)
    return np.asarray(
        [len(q_tok), fn_ratio, overlap_qs, overlap_qa, starts_wh, len(s_tok), len(a_tok)],
        dtype=np.float32,
    )


def build_ranker_training_data(
    real_questions: Sequence[str],
    real_articles: Sequence[str],
    real_answers: Sequence[str],
    n_negatives_per_pos: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Positives: real RACE questions. Negatives: shuffled / generated noise."""
    rng = np.random.default_rng(seed)
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for q, art, ans in zip(real_questions, real_articles, real_answers):
        if not q or not art:
            continue
        sentences = split_sentences(art)
        src = sentences[0] if sentences else ""
        X_rows.append(_fluency_features(q, src, ans))
        y_rows.append(1)

        for _ in range(n_negatives_per_pos):
            tokens = q.split()
            if len(tokens) > 2:
                idx = rng.permutation(len(tokens))
                noisy = " ".join([tokens[i] for i in idx])
            else:
                noisy = (q + " random words")[::-1]
            X_rows.append(_fluency_features(noisy, src, ans))
            y_rows.append(0)

    if not X_rows:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(X_rows), np.asarray(y_rows, dtype=np.int64)


def train_question_ranker(
    real_questions: Sequence[str],
    real_articles: Sequence[str],
    real_answers: Sequence[str],
    seed: int = 42,
) -> RandomForestClassifier:
    X, y = build_ranker_training_data(real_questions, real_articles, real_answers, seed=seed)
    if len(np.unique(y)) < 2:
        clf = RandomForestClassifier(n_estimators=10, random_state=seed)
        # fit on a tiny dummy so the model is callable
        X_dummy = np.zeros((2, 7), dtype=np.float32)
        y_dummy = np.array([0, 1], dtype=np.int64)
        clf.fit(X_dummy, y_dummy)
        return clf
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def rank_generated_questions(
    candidates: Sequence[GeneratedQuestion],
    answer: str,
    ranker: Optional[RandomForestClassifier] = None,
) -> List[GeneratedQuestion]:
    """Rank generated questions by ranker probability (or overlap fallback)."""
    if not candidates:
        return []
    if ranker is None:
        for c in candidates:
            c.score = _overlap_score(c.source_sentence, answer)
    else:
        feats = np.vstack(
            [_fluency_features(c.question_text, c.source_sentence, answer) for c in candidates]
        )
        try:
            proba = ranker.predict_proba(feats)[:, 1]
        except Exception:
            proba = ranker.predict(feats).astype(float)
        for c, p in zip(candidates, proba):
            c.score = float(p)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def generate_and_rank(
    article: str,
    answer: str,
    ranker: Optional[RandomForestClassifier] = None,
    top_k: int = 5,
) -> List[GeneratedQuestion]:
    cands = generate_questions(article, answer)
    ranked = rank_generated_questions(cands, answer=answer, ranker=ranker)
    return ranked[:top_k]
