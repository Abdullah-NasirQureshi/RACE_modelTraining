"""Tiny synthetic RACE-like dataset.

Used as a fallback when no Kaggle CSVs are present so the Streamlit UI is
always runnable, and as a fixture for the smoke tests in `tests/test_inference.py`.

Each sample mirrors RACE's column schema:
    id, article, question, A, B, C, D, answer
"""

from __future__ import annotations

from typing import List, Dict
import pandas as pd


_SAMPLES: List[Dict[str, str]] = [
    {
        "id": "demo_001",
        "article": (
            "The Silk Road was an ancient network of trade routes that connected "
            "the East and the West. For centuries, merchants travelled along it, "
            "trading silk, spices, tea, and precious stones. The route stretched "
            "from China through Central Asia to Europe. It also helped spread "
            "religions, technologies, and ideas between civilisations."
        ),
        "question": "What were merchants trading along the Silk Road?",
        "A": "Silk and spices",
        "B": "Cars and computers",
        "C": "Aeroplanes",
        "D": "Mobile phones",
        "answer": "A",
    },
    {
        "id": "demo_002",
        "article": (
            "Photosynthesis is the process by which green plants make their own "
            "food. They absorb sunlight using a green pigment called chlorophyll. "
            "During photosynthesis, plants take in carbon dioxide from the air "
            "and water from the soil, and they release oxygen as a by-product."
        ),
        "question": "Which gas do plants release during photosynthesis?",
        "A": "Carbon dioxide",
        "B": "Nitrogen",
        "C": "Oxygen",
        "D": "Hydrogen",
        "answer": "C",
    },
    {
        "id": "demo_003",
        "article": (
            "Mount Everest is the tallest mountain on Earth. It is located in "
            "the Himalayas on the border between Nepal and Tibet. Many climbers "
            "attempt to reach its summit each year, though the journey is "
            "extremely dangerous because of the cold, the thin air, and the "
            "risk of avalanches."
        ),
        "question": "Where is Mount Everest located?",
        "A": "Between India and China",
        "B": "Between Nepal and Tibet",
        "C": "In the Andes",
        "D": "In the Alps",
        "answer": "B",
    },
    {
        "id": "demo_004",
        "article": (
            "The human heart is a muscular organ about the size of a clenched "
            "fist. It pumps blood through the body's network of arteries and "
            "veins. The heart has four chambers: two atria and two ventricles. "
            "It beats around 100,000 times every day."
        ),
        "question": "How many chambers does the human heart have?",
        "A": "Two",
        "B": "Three",
        "C": "Four",
        "D": "Five",
        "answer": "C",
    },
    {
        "id": "demo_005",
        "article": (
            "William Shakespeare was an English playwright who lived in the "
            "sixteenth and early seventeenth centuries. He wrote thirty-seven "
            "plays and many sonnets. His most famous works include Hamlet, "
            "Macbeth, and Romeo and Juliet. His plays are still performed "
            "around the world today."
        ),
        "question": "What is William Shakespeare best known for?",
        "A": "Writing plays and sonnets",
        "B": "Painting portraits",
        "C": "Composing operas",
        "D": "Building cathedrals",
        "answer": "A",
    },
    {
        "id": "demo_006",
        "article": (
            "Recycling helps reduce waste and protect the environment. When "
            "people recycle paper, plastic, and metal, fewer raw materials need "
            "to be taken from nature. Recycling also saves energy and reduces "
            "pollution. Many cities now provide recycling bins to make it "
            "easier for households to participate."
        ),
        "question": "Why is recycling important?",
        "A": "It makes products more expensive",
        "B": "It reduces waste and protects the environment",
        "C": "It increases pollution",
        "D": "It uses more raw materials",
        "answer": "B",
    },
    {
        "id": "demo_007",
        "article": (
            "The Internet has transformed the way people communicate. Email, "
            "video calls, and social media allow friends and families to stay "
            "in touch across long distances. Businesses also use the Internet "
            "to reach customers around the world. However, too much screen "
            "time can affect health and sleep."
        ),
        "question": "What is one negative effect of using the Internet too much?",
        "A": "Faster communication",
        "B": "Wider customer base",
        "C": "Effects on health and sleep",
        "D": "Cheaper email",
        "answer": "C",
    },
    {
        "id": "demo_008",
        "article": (
            "Bees play a vital role in nature. They collect nectar from flowers "
            "and, in doing so, transfer pollen between plants. This process, "
            "called pollination, allows many crops and wild plants to reproduce. "
            "Without bees, the world's food supply would be at serious risk."
        ),
        "question": "What process do bees perform when they move pollen between flowers?",
        "A": "Photosynthesis",
        "B": "Pollination",
        "C": "Evaporation",
        "D": "Germination",
        "answer": "B",
    },
]


def load_demo_dataframe() -> pd.DataFrame:
    """Return a small DataFrame matching the RACE CSV schema."""
    return pd.DataFrame(_SAMPLES)


def get_demo_sample(idx: int = 0) -> Dict[str, str]:
    """Return a single demo sample by index (wraps around)."""
    return _SAMPLES[idx % len(_SAMPLES)].copy()
