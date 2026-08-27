from pathlib import Path

from quintropy import feedback_code, load_words
from quintropy.feedback import feedback_cache_path


def test_all_green():
    assert feedback_code("crash", "crash") == 242


def test_load_words_returns_complete_action_universe():
    words = load_words()
    assert len(words) == 12_972
    assert words[0].islower()


def test_duplicate_accounting_consistency():
    x = feedback_code("allee", "apple")
    assert isinstance(x, int)
    assert 0 <= x <= 242
    assert x == 167
    assert x == feedback_code("allee", "apple")


def test_feedback_cache_is_bound_to_the_ordered_word_universe():
    cache_dir = Path("cache")
    assert feedback_cache_path(cache_dir, ("cigar", "rebut")) != feedback_cache_path(
        cache_dir, ("rebut", "cigar")
    )
