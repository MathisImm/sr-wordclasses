"""Locations of the bundled resources and of the Zenodo data deposit."""
import os
from pathlib import Path

RESOURCES = Path(__file__).resolve().parents[2] / "resources"
FREQS = RESOURCES / "wikitext-103_train_freqs_lowercase.json"
LEXICON = RESOURCES / "pos_lexicon_wikitext-103_lowercase.json"


def deposit(*parts: str) -> Path:
    """Path inside the unpacked Zenodo deposit, located by $SRWC_DEPOSIT."""
    root = os.environ.get("SRWC_DEPOSIT")
    if not root:
        raise RuntimeError("Set SRWC_DEPOSIT to the unpacked Zenodo deposit "
                           "(the folder containing reproduce.py).")
    return Path(root).joinpath(*parts)
