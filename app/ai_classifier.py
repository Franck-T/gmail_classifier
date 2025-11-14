"""
ai_classifier.py
~~~~~~~~~~~~~~~~~

This module provides an AI‑powered classifier that assigns emails to one of
several categories using sentence embeddings and cosine similarity.  It
leverages a free, pretrained model from the `sentence-transformers` library
(``all-MiniLM-L6-v2``) to compute vector representations of text.  Each
category is described with a short natural language sentence; embeddings for
these descriptions are computed once at startup.  Emails are then assigned to
the category whose description has the highest cosine similarity with the
email’s own embedding.

Gmail’s default categories include **Primary**, **Promotions**, **Social**,
**Updates** and **Forums**【25304394913824†L40-L50】【617583219031976†L24-L35】.  We also add a custom **Work** category.  The
descriptions below summarise each category so that the embedding model can
capture their semantics.

This module can be imported as a library, or invoked as a CLI to classify a
single email or a batch of emails.  When run directly, it accepts JSON input
from stdin containing an array of objects with ``from``, ``subject`` and
``snippet`` fields and writes a JSON array of category strings to stdout.

Example usage as a module::

    from ai_classifier import classify_email
    category = classify_email(from_address, subject, snippet)

Command‑line usage for batch classification::

    echo '[{"from": "alice@example.com", "subject": "Sale now on", "snippet": "..."}]' | \\
        python ai_classifier.py
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


# Define the categories and their natural language descriptions.  These
# descriptions mirror Gmail’s definitions【25304394913824†L40-L50】【617583219031976†L24-L35】 and add a custom work category.
CATEGORY_DESCRIPTIONS = {
    "Primary": "Personal and important emails from people you know.",
    "Promotions": "Deals, offers, advertisements and other promotional emails.",
    "Social": "Messages from social networks and media‑sharing sites.",
    "Updates": "Automated confirmations, notifications, statements and reminders.",
    "Forums": "Messages from online groups, discussion boards and mailing lists.",
    "Work": "Work‑related emails from corporate or professional domains.",
}


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and return the sentence‑transformer model.  Cached to avoid reloading."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@lru_cache(maxsize=1)
def get_category_embeddings():
    """Compute embeddings for all category descriptions and cache them."""
    model = get_model()
    descriptions = list(CATEGORY_DESCRIPTIONS.values())
    embeddings = model.encode(descriptions, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings


def _compose_email_text(from_addr: Optional[str], subject: Optional[str], snippet: Optional[str]) -> str:
    """Concatenate relevant parts of the email into a single string for embedding."""
    parts = []
    if from_addr:
        parts.append(from_addr)
    if subject:
        parts.append(subject)
    if snippet:
        parts.append(snippet)
    return ". ".join(parts)


def classify_email(from_addr: Optional[str], subject: Optional[str], snippet: Optional[str]) -> str:
    """Classify a single email into a category based on its semantic similarity.

    Parameters
    ----------
    from_addr : str
        Sender's email address (may be None).
    subject : str
        Email subject line.
    snippet : str
        Short snippet of the message body.

    Returns
    -------
    str
        The category with the highest cosine similarity score.
    """
    model = get_model()
    categories = list(CATEGORY_DESCRIPTIONS.keys())
    cat_embeddings = get_category_embeddings()
    email_text = _compose_email_text(from_addr, subject, snippet)
    email_embedding = model.encode(email_text, convert_to_tensor=True, normalize_embeddings=True)
    # Compute cosine similarity between email and each category
    scores = util.cos_sim(email_embedding, cat_embeddings)[0]
    # Find the index of the highest score
    best_idx = int(np.argmax(scores.cpu().numpy()))
    return categories[best_idx]


def classify_batch(messages: Iterable[dict]) -> List[str]:
    """Classify a batch of messages.  Each message should be a dict with keys
    ``from``, ``subject`` and ``snippet``.  Returns a list of category strings.
    """
    model = get_model()
    categories = list(CATEGORY_DESCRIPTIONS.keys())
    cat_embeddings = get_category_embeddings()
    texts = []
    for msg in messages:
        texts.append(_compose_email_text(msg.get("from"), msg.get("subject"), msg.get("snippet")))
    email_embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(email_embeddings, cat_embeddings)
    # For each email, pick the category with the highest similarity
    best_indices = np.argmax(scores.cpu().numpy(), axis=1)
    return [categories[int(idx)] for idx in best_indices]


def main() -> None:
    """Read a JSON list of messages from stdin and output a JSON list of categories."""
    data = sys.stdin.read()
    if not data.strip():
        print("[]")
        return
    messages = json.loads(data)
    categories = classify_batch(messages)
    print(json.dumps(categories))


if __name__ == "__main__":
    main()