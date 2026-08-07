# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Hugging Face Hub object store.

Reading an ``hf://`` URL needs no store: :func:`vortex.io.read_url` resolves it, taking credentials
from ``HF_TOKEN`` or the saved login. :class:`HfStore` covers the two cases a URL cannot express — a
token held in a variable rather than the environment, and a read that must stay anonymous even
though the environment offers credentials.
"""

from __future__ import annotations

from vortex._lib import HfStore as HfStore

__all__ = ["HfStore"]
