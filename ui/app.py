"""Streamlit entrypoint for the RACE RC + Quiz Generation system.

Run with:
    streamlit run ui/app.py

Implements the four spec-required screens:
  1) Article Input
  2) Question & Answer Quiz View
  3) Hint Panel (graduated)
  4) Developer / Analytics Dashboard

The app gracefully falls back to the synthetic demo dataset (`src.demo_data`)
when no Kaggle CSVs are present and no trained models exist, so the UI is
always runnable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Make `src` importable when running `streamlit run ui/app.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.demo_data import load_demo_dataframe                                # noqa: E402
from src.inference import (                                                  # noqa: E402
    Artifacts,
    generate_distractors,
    generate_questions,
    load_artifacts,
    predict_best_option,
    predict_question_type,
    rank_hints,
    verify_answer,
)
from src.preprocessing import OPTION_LETTERS                                  # noqa: E402


# ---------------------------------------------------------------------------
# Page configuration (must be first st.* call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RACE RC + Quiz Generator",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained models...")
def get_artifacts() -> Artifacts:
    return load_artifacts(force_reload=True)


@st.cache_data(show_spinner=False)
def get_demo_df() -> pd.DataFrame:
    return load_demo_dataframe()


@st.cache_data(show_spinner=False)
def maybe_load_val_df() -> Optional[pd.DataFrame]:
    """Load data/raw/val.csv if present (for the 'random RACE sample' feature)."""
    path = os.path.join("data", "raw", "val.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Dict:
    path = os.path.join("models", "metrics.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def load_manifest_json() -> Dict:
    path = os.path.join("models", "manifest.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("article", "")
    ss.setdefault("question", "")
    ss.setdefault("options", {l: "" for l in OPTION_LETTERS})
    ss.setdefault("gold_answer", "")
    ss.setdefault("user_choice", None)
    ss.setdefault("checked", False)
    ss.setdefault("hints_shown", set())          # which hint expanders have been opened
    ss.setdefault("history", [])                 # list of dict logs (for analytics)
    ss.setdefault("generated_questions", [])
    ss.setdefault("auto_distractors", [])


def _set_sample(row: Dict) -> None:
    """Populate state from a RACE-like row (real or demo)."""
    st.session_state.article = str(row.get("article", ""))
    st.session_state.question = str(row.get("question", ""))
    gold = str(row.get("answer", "")).strip().upper()
    st.session_state.gold_answer = gold
    st.session_state.options = {
        letter: str(row.get(letter, "")) for letter in OPTION_LETTERS
    }
    st.session_state.user_choice = None
    st.session_state.checked = False
    st.session_state.hints_shown = set()
    st.session_state.generated_questions = []
    st.session_state.auto_distractors = []


def _log_event(kind: str, payload: Dict) -> None:
    st.session_state.history.append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            **payload,
        }
    )


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------

_init_state()
artifacts = get_artifacts()

with st.sidebar:
    st.title(":books: RACE RC + Quiz")
    st.caption("Classical-ML reading comprehension + quiz generator")

    page = st.radio(
        "Navigate",
        [
            "1. Article Input",
            "2. Quiz View",
            "3. Hint Panel",
            "4. Analytics Dashboard",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**Model status**")
    available = artifacts.available
    rows = [
        ("Vectorizer", available.get("vectorizer", False)),
        ("LR verifier", available.get("lr_verifier", False)),
        ("SVM verifier", available.get("svm_verifier", False)),
        ("Ensemble", available.get("ensemble", False)),
        ("NB question-type", available.get("nb_qtype", False)),
        ("Distractor LR", available.get("distractor_lr", False)),
        ("Hint LR", available.get("hint_lr", False)),
    ]
    for name, ok in rows:
        st.markdown(f"- {name}: {'OK' if ok else 'missing'}")

    if not artifacts.is_trained:
        st.info(
            "No trained models found. The app is using rule-based fallbacks. "
            "Train with `python -m src.train_all --demo` for a quick test."
        )

    st.divider()
    if st.button("Reload artifacts", use_container_width=True):
        get_artifacts.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Screen 1: Article Input
# ---------------------------------------------------------------------------

def render_article_input() -> None:
    st.header("1. Article Input")
    st.caption(
        "Paste a passage, upload a `.txt`, or load a sample. The system will "
        "run **Model A** (verify the chosen answer) and **Model B** "
        "(generate distractors and graduated hints)."
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Passage and question")
        article = st.text_area(
            "Reading passage",
            value=st.session_state.article,
            height=240,
            placeholder="Paste a reading passage here ...",
            key="article_input",
        )
        question = st.text_input(
            "Question (optional - leave blank to auto-generate)",
            value=st.session_state.question,
            placeholder="Type a question, or leave blank to use Model A's generator",
            key="question_input",
        )

        st.markdown("**Options (A / B / C / D)** - all four are required for the quiz screen.")
        oc1, oc2 = st.columns(2)
        opts: Dict[str, str] = {}
        with oc1:
            opts["A"] = st.text_input("Option A", value=st.session_state.options.get("A", ""), key="opt_A")
            opts["C"] = st.text_input("Option C", value=st.session_state.options.get("C", ""), key="opt_C")
        with oc2:
            opts["B"] = st.text_input("Option B", value=st.session_state.options.get("B", ""), key="opt_B")
            opts["D"] = st.text_input("Option D", value=st.session_state.options.get("D", ""), key="opt_D")

        gold = st.selectbox(
            "Gold answer letter (optional)",
            options=[""] + list(OPTION_LETTERS),
            index=([""] + list(OPTION_LETTERS)).index(st.session_state.gold_answer)
                  if st.session_state.gold_answer in OPTION_LETTERS else 0,
            help="If you know the correct option, set it here so Model B can use it as the answer key.",
        )

    with col_right:
        st.subheader("Quick load")
        st.caption("Try the system instantly with a sample.")

        val_df = maybe_load_val_df()
        sample_source = "Demo set"
        if val_df is not None:
            sample_source = st.selectbox(
                "Sample source",
                ["Demo set", "RACE val.csv (random)"],
                index=0,
            )

        if st.button("Load random sample", use_container_width=True):
            if sample_source == "RACE val.csv (random)" and val_df is not None:
                row = val_df.sample(1).iloc[0].to_dict()
            else:
                row = get_demo_df().sample(1).iloc[0].to_dict()
            _set_sample(row)
            st.rerun()

        st.divider()
        st.subheader("Upload .txt")
        up = st.file_uploader("Upload a passage", type=["txt"])
        if up is not None:
            try:
                st.session_state.article = up.read().decode("utf-8", errors="ignore")
                st.success("Passage loaded from file. You can now type the question above.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.divider()
    submit = st.button(
        "Submit -> run Model A + Model B",
        type="primary",
        use_container_width=True,
    )

    if submit:
        if not article.strip():
            st.warning("Please paste a passage first.")
            return
        st.session_state.article = article
        st.session_state.question = question
        st.session_state.options = opts
        st.session_state.gold_answer = gold
        st.session_state.user_choice = None
        st.session_state.checked = False
        st.session_state.hints_shown = set()

        gold_text = opts.get(gold, "") if gold else ""

        # Pre-compute hints + distractors so the other tabs feel snappy
        if not gold_text:
            # If no gold answer specified, treat the longest option as the
            # 'answer' for distractor/hint generation (best-effort).
            non_empty = [(k, v) for k, v in opts.items() if v.strip()]
            if non_empty:
                gold_text = max(non_empty, key=lambda kv: len(kv[1]))[1]

        with st.spinner("Generating distractors and hints..."):
            try:
                st.session_state.auto_distractors = generate_distractors(
                    article, gold_text or article[:50], artifacts, n_distractors=3
                )
            except Exception as e:
                st.session_state.auto_distractors = []
                st.error(f"Distractor generation failed: {e}")

            if not question.strip() and gold_text:
                try:
                    st.session_state.generated_questions = generate_questions(
                        article, gold_text, artifacts, top_k=3
                    )
                    if st.session_state.generated_questions:
                        st.session_state.question = (
                            st.session_state.generated_questions[0].question_text
                        )
                except Exception as e:
                    st.error(f"Question generation failed: {e}")

        _log_event(
            "submit",
            {
                "article_len": len(article),
                "has_gold": bool(gold),
                "has_question": bool(question.strip()),
            },
        )
        st.success("Done. Open the **Quiz View** or **Hint Panel** in the sidebar.")


# ---------------------------------------------------------------------------
# Screen 2: Quiz View
# ---------------------------------------------------------------------------

def render_quiz_view() -> None:
    st.header("2. Quiz View")

    if not st.session_state.article:
        st.info("No passage loaded yet. Go to **Article Input** and submit one first.")
        return

    article = st.session_state.article
    question = st.session_state.question or "(no question yet)"

    with st.expander("Show reading passage", expanded=False):
        st.write(article)

    st.subheader("Question")
    st.markdown(f"**{question}**")

    qtype = predict_question_type(question, artifacts)
    st.caption(f"Question type (Model A NB head): `{qtype}`")

    st.subheader("Options")
    options = st.session_state.options
    # If the user did not supply 4 options but Model B has distractors,
    # use them as a fallback so the quiz is still playable.
    auto_d = st.session_state.auto_distractors
    if sum(1 for v in options.values() if v.strip()) < 4 and auto_d:
        options = options.copy()
        gold = st.session_state.gold_answer or "A"
        if gold not in OPTION_LETTERS:
            gold = "A"
        # Make sure the gold letter has the gold text, then pour distractors
        # into the remaining slots.
        gold_text = options.get(gold, "") or " ".join(auto_d[:1])
        options[gold] = gold_text
        non_gold = [l for l in OPTION_LETTERS if l != gold]
        for letter, dtext in zip(non_gold, auto_d):
            if not options.get(letter, "").strip():
                options[letter] = dtext

    chosen = st.session_state.user_choice
    cols = st.columns(4)
    for i, letter in enumerate(OPTION_LETTERS):
        with cols[i]:
            text = options.get(letter, "")
            disabled = not text.strip()
            label = f"{letter}) {text}" if text else f"{letter}) (empty)"
            if st.button(label, key=f"opt_btn_{letter}",
                         use_container_width=True, disabled=disabled,
                         type="primary" if chosen == letter else "secondary"):
                st.session_state.user_choice = letter
                st.session_state.checked = False
                st.rerun()

    st.divider()
    check = st.button("Check answer with Model A verifier",
                      type="primary", use_container_width=True,
                      disabled=st.session_state.user_choice is None)
    if check:
        st.session_state.checked = True
        chosen_letter = st.session_state.user_choice
        chosen_text = options.get(chosen_letter, "")
        with st.spinner("Verifying with Model A..."):
            best, scores, lat = predict_best_option(article, question, options, artifacts)
            chosen_p, _ = verify_answer(article, question, chosen_text, artifacts)

        _log_event(
            "verify",
            {
                "chosen": chosen_letter,
                "model_best": best,
                "chosen_proba": float(chosen_p),
                "latency_ms": float(lat),
                "gold": st.session_state.gold_answer,
            },
        )

        gold = st.session_state.gold_answer
        if gold and gold in OPTION_LETTERS:
            if chosen_letter == gold:
                st.success(f"Correct! :white_check_mark:  Model A confidence: {chosen_p:.2f}")
            else:
                st.error(
                    f"Incorrect.  You picked **{chosen_letter}**, "
                    f"the correct answer is **{gold}**.  "
                    f"Model A confidence in your pick: {chosen_p:.2f}"
                )
        else:
            # No gold known; use the verifier's top-1 prediction.
            if chosen_letter == best:
                st.success(
                    f"Model A's top prediction matches yours (**{best}**). "
                    f"Confidence: {chosen_p:.2f}"
                )
            else:
                st.warning(
                    f"Model A would have picked **{best}** instead of **{chosen_letter}**.  "
                    f"Confidence in your pick: {chosen_p:.2f}"
                )

        with st.expander("Per-option Model A confidence", expanded=False):
            df_scores = pd.DataFrame(
                [{"option": k, "P(correct)": v} for k, v in scores.items()]
            ).sort_values("P(correct)", ascending=False)
            st.dataframe(df_scores, use_container_width=True, hide_index=True)
            st.caption(f"Verification latency: {lat:.1f} ms")


# ---------------------------------------------------------------------------
# Screen 3: Hint Panel
# ---------------------------------------------------------------------------

def render_hint_panel() -> None:
    st.header("3. Hint Panel")

    if not st.session_state.article:
        st.info("No passage loaded yet. Go to **Article Input** first.")
        return

    article = st.session_state.article
    question = st.session_state.question or ""
    gold = st.session_state.gold_answer
    gold_text = st.session_state.options.get(gold, "") if gold else ""
    if not gold_text:
        # fall back to the longest option
        non_empty = [(k, v) for k, v in st.session_state.options.items() if v.strip()]
        gold_text = max(non_empty, key=lambda kv: len(kv[1]))[1] if non_empty else ""

    with st.spinner("Ranking hints with Model B..."):
        hints = rank_hints(article, question, gold_text, artifacts, n_hints=3)

    if not hints:
        st.warning("Could not extract hint sentences from the passage.")
        return

    st.markdown(
        "Open the hints in order. Hint 1 is the most general; Hint 3 is the most specific. "
        "The **Reveal Answer** button unlocks only after you have opened all three."
    )

    for i, hint in enumerate(hints, start=1):
        with st.expander(f"Hint {i}  -  {'most general' if i == 1 else ('more specific' if i == 2 else 'near-explicit')}",
                         expanded=False):
            st.markdown(f"> {hint}")
            st.session_state.hints_shown.add(i)

    if i >= 3 and len(st.session_state.hints_shown) >= 3:
        if st.button("Reveal answer", type="primary", use_container_width=True):
            if gold and gold in OPTION_LETTERS:
                st.success(
                    f"Correct answer: **{gold}) {st.session_state.options.get(gold, '')}**"
                )
            elif gold_text:
                st.success(f"Best guess from Model A: **{gold_text}**")
            else:
                st.info("No gold answer was provided.")
            _log_event("reveal_answer", {"gold": gold})
    else:
        st.button("Reveal answer (open all hints first)", disabled=True,
                  use_container_width=True)


# ---------------------------------------------------------------------------
# Screen 4: Analytics Dashboard
# ---------------------------------------------------------------------------

def _confusion_matrix_figure(cm: List[List[int]], labels: List[str]):
    import plotly.express as px
    df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion matrix (verifier)",
    )
    return fig


def render_dashboard() -> None:
    st.header("4. Developer / Analytics Dashboard")

    metrics = load_metrics_json()
    manifest = load_manifest_json()

    if not metrics and not manifest:
        st.info(
            "No `models/metrics.json` or `models/manifest.json` found yet. "
            "Run `python -m src.train_all --demo` (or with real data) to populate this view."
        )
    else:
        st.subheader("Persisted evaluation metrics")
        if metrics:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Verifier accuracy", f"{metrics.get('verifier_accuracy', 0):.3f}")
            mc2.metric("Verifier macro-F1", f"{metrics.get('verifier_macro_f1', 0):.3f}")
            mc3.metric("Distractor F1", f"{metrics.get('distractor_f1', 0):.3f}")
            mc4.metric("Hint precision@1", f"{metrics.get('hint_precision_at_1', 0):.3f}")

            mc5, mc6, mc7, mc8 = st.columns(4)
            mc5.metric("Verifier EM", f"{metrics.get('verifier_exact_match', 0):.3f}")
            mc6.metric("Distractor precision", f"{metrics.get('distractor_precision', 0):.3f}")
            mc7.metric("Distractor recall", f"{metrics.get('distractor_recall', 0):.3f}")
            mc8.metric("Avg inference (ms)", f"{metrics.get('avg_inference_ms', 0):.1f}")

            cm = metrics.get("confusion_matrix")
            cm_labels = metrics.get("confusion_labels")
            if cm and cm_labels:
                st.plotly_chart(_confusion_matrix_figure(cm, cm_labels),
                                use_container_width=True)

        if manifest:
            with st.expander("Training run manifest", expanded=False):
                st.json(manifest)

    st.divider()
    st.subheader("Live session log")
    history = st.session_state.history
    if not history:
        st.caption("Use the Quiz View - your interactions will appear here.")
        return

    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if "latency_ms" in df_hist.columns:
        verify_rows = df_hist[df_hist["kind"] == "verify"]
        if not verify_rows.empty:
            cA, cB, cC = st.columns(3)
            cA.metric("Verifications run", f"{len(verify_rows)}")
            cB.metric("Avg latency (ms)", f"{verify_rows['latency_ms'].mean():.1f}")
            if "gold" in verify_rows.columns and "chosen" in verify_rows.columns:
                ok = verify_rows.apply(
                    lambda r: r.get("chosen") == r.get("gold")
                              if r.get("gold") in OPTION_LETTERS else None,
                    axis=1,
                )
                ok = ok.dropna()
                if len(ok) > 0:
                    cC.metric("User-correct rate", f"{(ok.mean() * 100):.1f}%")

    csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download session log CSV",
        data=csv_bytes,
        file_name="session_log.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page.startswith("1"):
    render_article_input()
elif page.startswith("2"):
    render_quiz_view()
elif page.startswith("3"):
    render_hint_panel()
else:
    render_dashboard()
