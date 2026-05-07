"""Smoke tests for the unified inference API.

These run on the synthetic demo dataset and do NOT require Kaggle CSVs.

Usage
-----
From the project root:

    python -m unittest tests.test_inference -v

You can also run end-to-end training first if you want non-trivial outputs:

    python -m src.train_all --demo
    python -m unittest tests.test_inference -v
"""

from __future__ import annotations

import os
import sys
import unittest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class InferenceSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.demo_data import load_demo_dataframe
        from src.inference import load_artifacts

        cls.df = load_demo_dataframe()
        cls.artifacts = load_artifacts(force_reload=True)

    def test_demo_dataset_shape(self) -> None:
        self.assertEqual(len(self.df), 8)
        for col in ("article", "question", "A", "B", "C", "D", "answer"):
            self.assertIn(col, self.df.columns)

    def test_verify_answer_returns_probability(self) -> None:
        from src.inference import verify_answer

        row = self.df.iloc[0]
        gold_letter = str(row["answer"])
        gold_text = str(row[gold_letter])
        p, latency_ms = verify_answer(
            row["article"], row["question"], gold_text, self.artifacts
        )
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertGreater(latency_ms, 0.0)

    def test_predict_best_option_letter(self) -> None:
        from src.inference import predict_best_option
        from src.preprocessing import OPTION_LETTERS

        row = self.df.iloc[0]
        options = {l: str(row[l]) for l in OPTION_LETTERS}
        best, scores, _ = predict_best_option(
            row["article"], row["question"], options, self.artifacts
        )
        self.assertIn(best, OPTION_LETTERS)
        for letter, p in scores.items():
            self.assertIn(letter, OPTION_LETTERS)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_question_type_predicts_a_known_label(self) -> None:
        from src.inference import predict_question_type
        from src.preprocessing import WH_LABELS

        out = predict_question_type("What were merchants trading?", self.artifacts)
        # Either a Wh-bucket from the spec or any string the NB head produced
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)
        # The fallback always returns a known WH label when no NB head exists
        if not self.artifacts.available.get("nb_qtype", False):
            self.assertIn(out, WH_LABELS)

    def test_distractor_generation_returns_three_strings(self) -> None:
        from src.inference import generate_distractors

        row = self.df.iloc[0]
        gold_letter = str(row["answer"])
        gold_text = str(row[gold_letter])
        ds = generate_distractors(row["article"], gold_text, self.artifacts, n_distractors=3)
        # Up to 3, all distinct from the answer
        self.assertLessEqual(len(ds), 3)
        for d in ds:
            self.assertNotEqual(d.lower().strip(), gold_text.lower().strip())

    def test_hint_ranking_returns_at_most_three(self) -> None:
        from src.inference import rank_hints

        row = self.df.iloc[0]
        gold_letter = str(row["answer"])
        gold_text = str(row[gold_letter])
        hints = rank_hints(row["article"], row["question"], gold_text, self.artifacts, n_hints=3)
        self.assertLessEqual(len(hints), 3)
        for h in hints:
            self.assertIsInstance(h, str)
            self.assertGreater(len(h), 0)

    def test_question_generation_returns_candidates(self) -> None:
        from src.inference import generate_questions

        row = self.df.iloc[0]
        gold_letter = str(row["answer"])
        gold_text = str(row[gold_letter])
        cands = generate_questions(row["article"], gold_text, self.artifacts, top_k=3)
        self.assertGreater(len(cands), 0)
        for c in cands:
            self.assertTrue(c.question_text)
            self.assertGreaterEqual(c.score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
