"""Rich-terminal walkthroughs, one per session.

Each `sessionNN_*.py` module is runnable on its own:

    uv run python -m kv_cache_variants.lessons.session01_naive_decoding

It imports the *actual* implementation from the sibling `kv_cache_variants`
modules (memory_calc, attention/mha.py, etc.) — no attention math is
reimplemented here, this package is a presentation layer only.
"""
