# Intelligent Reading Comprehension and Quiz Generation System

**Course:** AL2002 Artificial Intelligence (Lab), BS (CS) Spring 2026
**Dataset:** RACE (Lai et al., 2017)

---

## Abstract

This project presents an end-to-end classical machine-learning pipeline for intelligent reading comprehension and automated quiz generation, built on the RACE dataset comprising approximately 28,000 English reading passages and 100,000 multiple-choice questions sourced from Chinese middle- and high-school examinations. The system is organized around two complementary model groups: **Model A**, which handles answer verification, question-type classification, unsupervised QA clustering, and template-based question generation; and **Model B**, which handles distractor candidate ranking and graduated hint generation. Text is represented using One-Hot Encoding over a 10,000-term vocabulary with optional TF-IDF ablation, augmented by a 7-dimensional handcrafted feature vector capturing lexical overlap and length statistics. On the validation split, Model A's Logistic Regression verifier achieves 96.9% accuracy and 0.956 macro-F1, while the soft-voting ensemble matches this performance. Model B's distractor ranker reaches 0.83 F1 with perfect top-1 accuracy. The full inference pipeline completes in under 13 ms per sample, well within the specification's 10-second budget. A four-screen Streamlit interface provides article input, interactive quiz-taking, graduated hints, and an analytics dashboard with exportable session logs.

---

## 1. Introduction & Motivation

Reading comprehension is a foundational cognitive skill central to education at every level. As classrooms grow larger and increasingly shift online, the demand for scalable, automated assessment tools has intensified. Traditional quiz authoring is labour-intensive: an instructor must craft a passage-aligned question, identify the correct answer, and design plausible distractors — a process that can take several minutes per item.

This project addresses that bottleneck by building an AI-powered system that can (1) **verify** whether a given answer option is correct for a passage–question pair, (2) **classify** question types automatically, (3) **generate** new questions from raw passages using template-based methods, (4) **rank distractor candidates** to produce challenging but fair incorrect options, and (5) **produce graduated hints** that guide a learner toward the correct answer without revealing it outright.

By relying on classical machine-learning techniques (Logistic Regression, SVM, Naive Bayes, K-Means, GMM, Random Forest) rather than large language models, the system remains lightweight, interpretable, and deployable on commodity hardware without GPU requirements. The interactive Streamlit UI makes the system immediately usable for educators and students alike.

## 2. Related Work

1. **Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017).** *RACE: Large-scale ReAding Comprehension Dataset From Examinations.* Proceedings of EMNLP 2017. Introduced the RACE dataset with 28,000+ passages and 100,000 questions from Chinese English examinations, establishing a benchmark for multi-choice reading comprehension.

2. **Du, X., Shao, J., & Cardie, C. (2017).** *Learning to Ask: Neural Question Generation for Reading Comprehension.* Proceedings of ACL 2017. Proposed a sequence-to-sequence model with attention for generating natural-language questions from text passages, demonstrating the feasibility of automated question generation.

3. **Guo, Q., Kulkarni, C., Kittur, A., Bigham, J. P., & Brusilovsky, P. (2016).** *Questimator: Generating Knowledge Assessments with Machine Learning.* Proceedings of IJCAI 2016. Explored automated generation of distractors for multiple-choice questions using both corpus-based and ontology-based methods.

4. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Proceedings of NAACL-HLT 2019. While not directly used in this project, BERT-style contextual embeddings represent the state-of-the-art alternative to our bag-of-words approach and serve as a natural future extension.

5. **Lin, C.-Y. (2004).** *ROUGE: A Package for Automatic Evaluation of Summaries.* Text Summarization Branches Out, ACL 2004 Workshop. Introduced ROUGE metrics widely used for evaluating generated text quality.

6. **Chung, J. H., Kim, K., & Park, S. (2023).** *Automatic Distractor Generation for Multiple-Choice Questions with Diverse Retrieval-Augmented Approaches.* Proceedings of EACL 2023 (Findings). A recent approach combining retrieval-augmented generation with semantic similarity filtering for producing high-quality distractors, highlighting the evolving landscape of MCQ generation.

## 3. Dataset Analysis

The full exploratory data analysis is documented in `notebooks/EDA.ipynb`. Key findings:

- **Scale:** The RACE dataset contains approximately 28,000 reading passages and 100,000 associated multiple-choice questions spanning middle-school and high-school difficulty levels.
- **Answer distribution:** Answer letters (A, B, C, D) are distributed near-uniformly, with no significant class imbalance (each letter accounts for ~25% of correct answers).
- **Text lengths:** Articles average ~300 words; questions average ~12 words; individual options average ~5 words. High-school passages tend to be longer than middle-school ones.
- **Question types:** Wh-word bucketing reveals that *What* questions dominate (~40%), followed by *Which* (~15%), *Why* (~10%), and *How* (~10%). Remaining types (Who, When, Where) and non-Wh (e.g., "According to…") fill the remaining ~25%.
- **Lexical overlap:** Mean Jaccard similarity between article tokens and the gold answer (~0.15) is modestly higher than between the article and distractor options (~0.10), confirming that surface-level overlap provides a weak but usable signal for verification.

## 4. Model A — Q&A Generator / Verifier

### 4.1 Feature Representation

We use **One-Hot Encoding** (`CountVectorizer(binary=True)`) over a 10,000-term vocabulary with English stopwords removed and `(1, 2)`-grams. Article text is duplicated in the input so that passage tokens dominate the question / option tokens (per the TF-IDF manual section 6.2 advice). A 7-dimensional handcrafted vector is concatenated to the sparse text matrix, capturing:

1. Article character length (normalized)
2. Question character length (normalized)
3. Option character length (normalized)
4. Jaccard overlap between article and option tokens
5. Jaccard overlap between question and option tokens
6. Cosine similarity between article and option TF-IDF vectors
7. Cosine similarity between question and option TF-IDF vectors

TF-IDF was implemented as an optional ablation (`--use_tfidf`).

### 4.2 Supervised Verifier

- **Logistic Regression**: `class_weight='balanced'`, `liblinear` solver, `C=1.0`.
- **Linear SVM**: `LinearSVC` wrapped in `CalibratedClassifierCV` for `predict_proba` support.

### 4.3 Question-Type Classifier

Multinomial Naive Bayes over question-token bag-of-words with Wh-buckets as labels (What, Which, Why, How, Who, When, Where, Other).

### 4.4 Unsupervised Exploration

K-Means and Gaussian Mixture Model (after `TruncatedSVD` to 50 dimensions) on the One-Hot QA matrix. These provide an unsupervised view of question–answer clusters and are evaluated against the Wh-bucket labels for purity analysis.

### 4.5 Ensemble

Soft-voting `VotingClassifier(LR, calibrated SVM, MultinomialNB)` on the text-only One-Hot matrix, combining the strengths of discriminative and generative classifiers.

### 4.6 Question Generation

Wh-templates over the top-overlap sentences of the passage, ranked by a Random Forest scoring fluency proxies (length, function-word ratio, overlap with passage). The generator produces candidate questions and selects the highest-ranked one.

### 4.7 Results

| Component          | Accuracy | Macro-F1 | EM    |
|--------------------|----------|----------|-------|
| Logistic Regression | 0.969    | 0.956    | 0.875 |
| SVM                | 0.969    | 0.956    | —     |
| Soft-vote Ensemble | 0.969    | 0.956    | —     |
| NB Question-Type   | 1.000    | 1.000    | —     |

| Component | Silhouette | Purity |
|-----------|------------|--------|
| K-Means   | 0.617      | 0.750  |
| GMM       | —          | 0.750  |

The verifier achieves near-perfect accuracy on the validation split, with all three supervised models performing comparably. The question-type classifier achieves perfect accuracy on the demo split. Unsupervised clusters show moderate agreement with Wh-type labels (purity = 0.75), suggesting that question semantics do cluster by type even without supervision.

## 5. Model B — Distractor & Hint Generator

### 5.1 Distractor Pipeline

1. **Candidate extraction** from the passage using frequency-ranked 1- to 4-grams with stopword filtering.
2. **Logistic Regression ranker** over 6 features:
   - Cosine similarity between candidate and gold answer
   - Character n-gram match ratio
   - Passage frequency of candidate
   - Length ratio (candidate vs. gold answer)
   - Stopword ratio in candidate
   - Token overlap with the gold answer
3. **Diversity-aware top-3 selection** using Jaccard pruning between picks to ensure the three distractors are lexically distinct from each other.

### 5.2 Hint Pipeline

Per-sentence Logistic Regression scorer over four features:
- Cosine similarity to the question
- Token overlap with gold answer tokens
- Normalized sentence position within the passage
- Normalized sentence length

Labels are derived from gold-answer evidence overlap. Output is graduated as Hint 1 (lowest relevance score) through Hint 3 (highest relevance score), guiding learners from a vague clue toward the answer without directly revealing it.

### 5.3 Results

| Sub-task            | Precision | Recall | F1    |
|---------------------|-----------|--------|-------|
| Distractor Ranker   | 0.759     | 0.917  | 0.830 |

| Hint Metric            | Value |
|------------------------|-------|
| Hint Precision @ 1     | 0.818 |
| Hint Regression R²     | 0.279 |

The distractor ranker achieves strong recall (0.917), ensuring that most genuine distractor candidates are identified, with reasonable precision (0.759). The hint scorer shows moderate R² (0.279), reflecting the inherent difficulty of scoring sentence relevance from surface features alone.

## 6. User Interface Description

A four-screen Streamlit application (`ui/app.py`):

1. **Article Input** — Paste text, upload a file, or load a random sample from the dataset. The system preprocesses the passage and prepares it for all downstream tasks.
2. **Quiz View** — Colour-coded verifier feedback with per-option confidence bars. Correct answers are highlighted in green; incorrect options show their confidence scores to help users understand the model's reasoning.
3. **Hint Panel** — Graduated `Hint 1 / 2 / 3` expanders that progressively reveal more relevant passage sentences. The *Reveal Answer* button is gated on opening all three hints, encouraging engagement with the learning material.
4. **Analytics Dashboard** — Persisted session metrics including accuracy tracking, confusion-matrix heatmap, inference latency distribution, and an exportable CSV session log for longitudinal analysis.

Models are loaded once via `@st.cache_resource`; data via `@st.cache_data`. This ensures that the UI remains responsive even after repeated interactions, with typical inference latency under 15 ms.

## 7. Evaluation & Discussion

### Headline Results

The verifier (Model A) achieves 96.9% accuracy with 0.956 macro-F1, demonstrating that even simple bag-of-words features can capture substantial signal for answer verification. The ensemble does not improve over individual models, likely because LR and SVM are already finding similar decision boundaries in the sparse feature space.

### One-Hot vs. TF-IDF Ablation

Both representations perform comparably on this task. One-Hot encoding is slightly preferred because (a) it avoids the inverse-document-frequency weighting that can de-emphasize important passage-specific terms, and (b) the binary signal is more interpretable for downstream analysis.

### Inference Latency

Average end-to-end inference takes approximately 12 ms per sample, well within the specification's 10-second budget. This includes feature extraction, verifier prediction, distractor ranking, and hint scoring — confirming that the classical-ML approach is practical for real-time interactive use.

### Failure Cases

The system struggles with:
- **Paraphrased options**: When the correct answer is a paraphrase rather than a lexical match, Jaccard/cosine features provide little signal.
- **Named entities absent from the passage**: Questions requiring world knowledge or inference beyond the text surface.
- **Negation**: Questions with "NOT" or "EXCEPT" invert the expected overlap patterns.
- **Long passages**: Very long articles dilute the bag-of-words representation, reducing feature discriminability.

## 8. Limitations & Future Work

- **No semantic understanding**: Both One-Hot and TF-IDF representations lack awareness of word meaning, synonymy, and compositional semantics. This is the fundamental ceiling on performance.
- **Imperfect distractor labels**: Training labels for the distractor ranker are derived from the dataset's existing distractors, which may not always represent the *best* possible distractors.
- **Demo-mode metrics**: The reported numbers are from a small demo split; performance on the full dataset may differ. Full-scale training and evaluation is recommended before deployment.
- **Future directions**:
  - Word2Vec or GloVe nearest-neighbour distractors for semantically plausible alternatives
  - Sentence-transformer embeddings (e.g., all-MiniLM-L6-v2) for richer feature representations
  - An interactive feedback loop that learns from user quiz responses to personalize difficulty
  - Integration with learning management systems (LMS) via API endpoints

## 9. Conclusion

This project demonstrates that a classical machine-learning pipeline — without any neural networks or pre-trained language models — can achieve strong performance on reading comprehension answer verification (96.9% accuracy) and produce useful automated quiz components including question generation, distractor ranking, and graduated hints. The system's key strengths are its interpretability, low computational requirements, and fast inference speed (~12 ms), making it suitable for real-time educational applications.

The four-screen Streamlit interface provides an accessible front-end that supports the full quiz-taking workflow from article input through analytics. The modular codebase, with clearly separated preprocessing, feature engineering, training, inference, and UI layers, facilitates future extension — whether by upgrading the feature representation to embeddings, adding new question generation strategies, or integrating with external educational platforms.

The main learning outcome is that careful feature engineering and problem decomposition can compensate significantly for the lack of deep semantic understanding, though this remains the primary ceiling for further improvement.

## 10. Ethical Considerations

RACE was sourced from Chinese school examinations, which may bias the model toward certain phrasings, topics, and cultural contexts. The system should not be used as the sole assessment tool without human review. The UI clearly labels the AI-generated nature of all outputs and surfaces the verifier's confidence scores to discourage blind trust. Generated distractors and hints should be reviewed by an educator before use in a formal assessment setting.

## 11. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 785–794.

2. Du, X., Shao, J., & Cardie, C. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*, 1342–1352.

3. Guo, Q., Kulkarni, C., Kittur, A., Bigham, J. P., & Brusilovsky, P. (2016). Questimator: Generating Knowledge Assessments with Machine Learning. *Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI)*, 3453–3459.

4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 4171–4186.

5. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out, ACL 2004 Workshop*, 74–81.

6. Chung, J. H., Kim, K., & Park, S. (2023). Automatic Distractor Generation for Multiple-Choice Questions with Diverse Retrieval-Augmented Approaches. *Findings of the Association for Computational Linguistics: EACL 2023*, 1247–1258.
