"""Model A training: verifier, NB question-type, clustering, ensemble.

Outputs (under models/model_a/traditional/):
    vectorizer.pkl              CountVectorizer(binary=True) or TfidfVectorizer
    lr_verifier.pkl             Logistic Regression
    svm_verifier.pkl            CalibratedClassifierCV(LinearSVC)
    nb_qtype.pkl                MultinomialNB on question tokens
    nb_qtype_vec.pkl            CountVectorizer used by the NB head
    kmeans.pkl                  K-Means on One-Hot QA features
    gmm.pkl                     GaussianMixture (on a SVD reduction)
    gmm_svd.pkl                 TruncatedSVD used to dense-ify input for GMM
    ensemble.pkl                Soft-voting LR + SVM + NB
    qgen_ranker.pkl             Random Forest ranker for generated questions
    metrics_a.json              all Model A metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .features import (
    build_onehot_vectorizer,
    build_tfidf_vectorizer,
    build_verifier_features,
)
from .preprocessing import (
    attach_question_type,
    clean_text,
    combined_verifier_text,
    expand_to_verifier_examples,
    WH_LABELS,
)
from .question_generation import train_question_ranker


MODEL_A_DIR = os.path.join("models", "model_a", "traditional")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard cluster purity: max-overlap label / N."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    correct = (
        df.groupby("pred")["true"]
        .agg(lambda s: s.value_counts().iloc[0])
        .sum()
    )
    return float(correct) / float(len(y_true)) if len(y_true) else 0.0


@dataclass
class ModelAMetrics:
    verifier_lr_accuracy: float
    verifier_lr_macro_f1: float
    verifier_lr_em: float
    verifier_svm_accuracy: float
    verifier_svm_macro_f1: float
    verifier_ensemble_accuracy: float
    verifier_ensemble_macro_f1: float
    qtype_nb_accuracy: float
    qtype_nb_macro_f1: float
    kmeans_silhouette: float
    kmeans_purity: float
    gmm_purity: float
    n_train_examples: int
    n_val_examples: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Exact Match metric for the verifier (per-question correctness)
# ---------------------------------------------------------------------------

def per_question_em(
    df_examples: pd.DataFrame,
    proba_correct: np.ndarray,
    gold_letters: pd.Series,
) -> float:
    """Treat each (article, question) group as a 4-way MCQ and check
    whether the highest-scoring option matches the gold letter.

    df_examples has a `source_idx` column linking the 4 expanded rows back to
    the original RACE row, plus `option_letter`.
    """
    df = df_examples.copy()
    df["proba"] = proba_correct
    correct = 0
    total = 0
    for src_idx, group in df.groupby("source_idx"):
        if len(group) == 0:
            continue
        best = group.loc[group["proba"].idxmax()]
        gold = gold_letters.iloc[int(src_idx)]
        if str(best["option_letter"]).strip().upper() == str(gold).strip().upper():
            correct += 1
        total += 1
    return float(correct) / float(total) if total else 0.0


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_model_a(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    use_tfidf: bool = False,
    n_clusters: int = 6,
    seed: int = 42,
    output_dir: str = MODEL_A_DIR,
) -> ModelAMetrics:
    _ensure_dir(output_dir)

    train_df = attach_question_type(train_df)
    val_df = attach_question_type(val_df)

    train_examples, y_train = expand_to_verifier_examples(train_df)
    val_examples, y_val = expand_to_verifier_examples(val_df)

    print(
        f"[Model A] verifier examples: train={len(train_examples)}  val={len(val_examples)}"
    )

    # ---- Vectorizer (One-Hot is primary; TF-IDF only via --use_tfidf) ------
    train_corpus = [
        combined_verifier_text(a, q, o)
        for a, q, o in zip(
            train_examples["article"],
            train_examples["question"],
            train_examples["option_text"],
        )
    ]
    if use_tfidf:
        print("[Model A] using TF-IDF features (ablation)")
        vectorizer = build_tfidf_vectorizer(train_corpus)
    else:
        print("[Model A] using One-Hot Encoding features (binary CountVectorizer)")
        vectorizer = build_onehot_vectorizer(train_corpus)

    X_train = build_verifier_features(train_examples, vectorizer)
    X_val = build_verifier_features(val_examples, vectorizer)

    # ---- Logistic Regression verifier --------------------------------------
    print("[Model A] training Logistic Regression verifier ...")
    lr = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=seed,
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    lr_acc = float(accuracy_score(y_val, lr_pred))
    lr_f1 = float(f1_score(y_val, lr_pred, average="macro", zero_division=0))
    lr_em = per_question_em(val_examples, lr_proba, val_df["answer"])
    print(f"   LR  acc={lr_acc:.3f}  macroF1={lr_f1:.3f}  EM={lr_em:.3f}")

    # ---- Linear SVM verifier (calibrated for predict_proba) ----------------
    print("[Model A] training Linear SVM verifier (calibrated) ...")
    base_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    svm = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_val)
    svm_acc = float(accuracy_score(y_val, svm_pred))
    svm_f1 = float(f1_score(y_val, svm_pred, average="macro", zero_division=0))
    print(f"   SVM acc={svm_acc:.3f}  macroF1={svm_f1:.3f}")

    # ---- Multinomial NB question-type classifier (spec 4.2.1) --------------
    print("[Model A] training Naive Bayes question-type classifier ...")
    nb_vec = CountVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 1))
    nb_X_train = nb_vec.fit_transform(train_df["question"].astype(str))
    nb_y_train = train_df["q_type"].astype(str).values
    nb_X_val = nb_vec.transform(val_df["question"].astype(str))
    nb_y_val = val_df["q_type"].astype(str).values
    nb = MultinomialNB()
    nb.fit(nb_X_train, nb_y_train)
    nb_pred = nb.predict(nb_X_val)
    nb_acc = float(accuracy_score(nb_y_val, nb_pred))
    nb_f1 = float(f1_score(nb_y_val, nb_pred, average="macro", zero_division=0))
    print(f"   NB  acc={nb_acc:.3f}  macroF1={nb_f1:.3f}")

    # ---- Ensemble (soft voting LR + SVM + NB on verifier features) ---------
    # NB takes a different feature space, so we wrap a clean MultinomialNB
    # on the *binary text matrix only* of verifier examples.
    print("[Model A] training ensemble (soft voting LR + SVM + NB) ...")
    text_only_train = vectorizer.transform(train_corpus)
    text_only_val = vectorizer.transform(
        [
            combined_verifier_text(a, q, o)
            for a, q, o in zip(
                val_examples["article"],
                val_examples["question"],
                val_examples["option_text"],
            )
        ]
    )

    nb_verifier = MultinomialNB()
    # NB cannot handle negatives; CountVectorizer(binary=True) and TfidfVectorizer
    # are both >= 0, so we are safe.
    nb_verifier.fit(text_only_train, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=1.0, solver="liblinear",
                                      class_weight="balanced", max_iter=200,
                                      random_state=seed)),
            ("svm", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
                cv=3, method="sigmoid")),
            ("nb", MultinomialNB()),
        ],
        voting="soft",
        n_jobs=None,
    )
    # Ensemble trained on the text-only matrix so all 3 estimators share the
    # same feature space (NB requires non-negative inputs).
    ensemble.fit(text_only_train, y_train)
    ens_pred = ensemble.predict(text_only_val)
    ens_acc = float(accuracy_score(y_val, ens_pred))
    ens_f1 = float(f1_score(y_val, ens_pred, average="macro", zero_division=0))
    print(f"   ENS acc={ens_acc:.3f}  macroF1={ens_f1:.3f}")

    # ---- K-Means clustering (spec 4.2.2) ----------------------------------
    print("[Model A] running K-Means clustering ...")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    km_labels = km.fit_predict(text_only_train)
    # Silhouette on a sample (full RACE is too large)
    sample_size = min(2000, text_only_train.shape[0])
    if sample_size > n_clusters + 1:
        sample_idx = np.random.default_rng(seed).choice(
            text_only_train.shape[0], size=sample_size, replace=False
        )
        try:
            km_sil = float(
                silhouette_score(text_only_train[sample_idx], km_labels[sample_idx])
            )
        except Exception:
            km_sil = 0.0
    else:
        km_sil = 0.0

    # Map clusters to question types and compute purity
    qtype_per_example = []
    for src_idx in train_examples["source_idx"].values:
        qtype_per_example.append(train_df["q_type"].iloc[int(src_idx)])
    km_purity = _purity_score(np.asarray(qtype_per_example), km_labels)
    print(f"   KM  silhouette={km_sil:.3f}  purity={km_purity:.3f}")

    # ---- GMM (spec 4.2.2) -------------------------------------------------
    print("[Model A] running Gaussian Mixture (after TruncatedSVD) ...")
    svd_components = min(50, max(2, text_only_train.shape[1] - 1))
    gmm_svd = TruncatedSVD(n_components=svd_components, random_state=seed)
    # Fit on a sample for speed
    svd_sample = min(20_000, text_only_train.shape[0])
    sample_idx = np.random.default_rng(seed).choice(
        text_only_train.shape[0], size=svd_sample, replace=False
    )
    gmm_svd.fit(text_only_train[sample_idx])
    X_train_dense = gmm_svd.transform(text_only_train)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        max_iter=50,
        random_state=seed,
    )
    gmm.fit(X_train_dense)
    gmm_labels = gmm.predict(X_train_dense)
    gmm_purity = _purity_score(np.asarray(qtype_per_example), gmm_labels)
    print(f"   GMM purity={gmm_purity:.3f}")

    # ---- Question-generation ranker --------------------------------------
    print("[Model A] training Random Forest ranker for question generation ...")
    qgen_ranker = train_question_ranker(
        real_questions=train_df["question"].astype(str).tolist(),
        real_articles=train_df["article"].astype(str).tolist(),
        real_answers=[
            str(row[row["answer"]]) if row["answer"] in ("A", "B", "C", "D") else ""
            for _, row in train_df.iterrows()
        ],
        seed=seed,
    )

    # ---- Persist artifacts -----------------------------------------------
    print(f"[Model A] saving artifacts to {output_dir} ...")
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
    joblib.dump(lr, os.path.join(output_dir, "lr_verifier.pkl"))
    joblib.dump(svm, os.path.join(output_dir, "svm_verifier.pkl"))
    joblib.dump(nb, os.path.join(output_dir, "nb_qtype.pkl"))
    joblib.dump(nb_vec, os.path.join(output_dir, "nb_qtype_vec.pkl"))
    joblib.dump(km, os.path.join(output_dir, "kmeans.pkl"))
    joblib.dump(gmm, os.path.join(output_dir, "gmm.pkl"))
    joblib.dump(gmm_svd, os.path.join(output_dir, "gmm_svd.pkl"))
    joblib.dump(ensemble, os.path.join(output_dir, "ensemble.pkl"))
    joblib.dump(qgen_ranker, os.path.join(output_dir, "qgen_ranker.pkl"))

    metrics = ModelAMetrics(
        verifier_lr_accuracy=lr_acc,
        verifier_lr_macro_f1=lr_f1,
        verifier_lr_em=lr_em,
        verifier_svm_accuracy=svm_acc,
        verifier_svm_macro_f1=svm_f1,
        verifier_ensemble_accuracy=ens_acc,
        verifier_ensemble_macro_f1=ens_f1,
        qtype_nb_accuracy=nb_acc,
        qtype_nb_macro_f1=nb_f1,
        kmeans_silhouette=km_sil,
        kmeans_purity=km_purity,
        gmm_purity=gmm_purity,
        n_train_examples=int(len(train_examples)),
        n_val_examples=int(len(val_examples)),
    )
    with open(os.path.join(output_dir, "metrics_a.json"), "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print("[Model A] done.")
    return metrics
