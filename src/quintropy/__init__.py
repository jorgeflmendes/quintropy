"""Quintropy's reproducible five-letter puzzle research toolkit."""

from .data import load_allowed_words, load_answer_words, load_history
from .feedback import feedback_code
from .policy import EntropyPolicy, PolicyConfig

load_words = load_allowed_words

__all__ = [
    "EntropyPolicy",
    "PolicyConfig",
    "feedback_code",
    "load_allowed_words",
    "load_answer_words",
    "load_history",
    "load_words",
]
