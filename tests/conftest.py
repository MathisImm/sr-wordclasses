import os

import pytest


needs_deposit = pytest.mark.skipif(
    not os.environ.get("SRWC_DEPOSIT"), reason="set SRWC_DEPOSIT to the unpacked Zenodo deposit")

# Equivalence tests import the original research code, which is not public; they
# document how each module was checked and run only where that code is available.
needs_original = pytest.mark.skipif(
    not os.environ.get("SRWC_ORIGINAL"), reason="set SRWC_ORIGINAL to the original research code")
