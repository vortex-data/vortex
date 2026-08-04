==================
Hugging Face Hub
==================

Vortex reads Hugging Face Hub repositories over ``hf://`` URLs. A Hub repository is a set of files
behind an HTTP endpoint that honours range requests, so no cloud SDK is involved and no extra build
feature is needed.

.. list-table::
   :header-rows: 1

   * - URL
     - Repository kind
   * - ``hf://datasets/<owner>/<name>[@<revision>][/<path>]``
     - Dataset
   * - ``hf://spaces/<owner>/<name>[@<revision>][/<path>]``
     - Space
   * - ``hf://<owner>/<name>[@<revision>][/<path>]``
     - Model

``<revision>`` is a branch, tag or commit, defaulting to ``main``. A revision containing ``/`` must
be percent-encoded, e.g. ``hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex``.

Configuration comes from the same environment variables ``huggingface_hub`` reads:

.. list-table::
   :header-rows: 1

   * - Variable
     - Meaning
   * - ``HF_TOKEN``
     - API token for private and gated repositories. Falls back to the token file at
       ``HF_TOKEN_PATH``, then ``$HF_HOME/token``, then ``$HOME/.cache/huggingface/token``.
   * - ``HF_ENDPOINT``
     - Hub endpoint, defaulting to ``https://huggingface.co``.

Reading from the Hub
====================

Pass an ``hf://`` URL directly. Public repositories need no credentials; private and gated ones
authenticate from ``HF_TOKEN`` or the saved login:

.. code-block:: python

   import vortex as vx

   vxf = vx.open("hf://datasets/org/name/data/train.vortex")
   for batch in vxf.to_arrow():
       ...

:class:`vortex.store.HfStore`
=============================

.. py:class:: vortex.store.HfStore(repo_id, *, repo_type="dataset", revision=None, token=None, endpoint=None)

   A Hugging Face Hub object store, rooted at one repository and revision.

   A URL is enough for most reads, so reach for this class only for the two things a URL cannot
   express: a token held in a variable rather than the environment, and a read that must stay
   anonymous even though the environment offers credentials.

   Because the store is rooted at the repository and revision, the path passed alongside it is a
   path *within* the repository.

   :param repo_id: The repository, as ``"<owner>/<name>"``.
   :param repo_type: ``"dataset"``, ``"model"`` or ``"space"``. Defaults to ``"dataset"``.
   :param revision: A branch, tag or commit. Defaults to ``main``. Unlike in a URL, a revision
       containing ``/`` is passed literally — the store percent-encodes it.
   :param token: ``None`` (the default) or ``True`` authenticates from ``HF_TOKEN`` or the saved
       login; ``False`` forces an anonymous read even when credentials are available; a string is
       used as the token directly.
   :param endpoint: Hub endpoint. Defaults to ``HF_ENDPOINT``, then ``https://huggingface.co``.

.. code-block:: python

   import vortex as vx
   from vortex.store import HfStore

   store = HfStore("org/name", revision="refs/convert/parquet", token="hf_...")

   # With `store=`, the path is a path within the repository.
   vxf = vx.open("data/train.vortex", store=store)

Listing
=======

The Hub does not implement WebDAV ``PROPFIND``, which is how object-store HTTP listing works, so a
Hub store cannot list a prefix. Opening a known path works, since that is a ``HEAD`` plus ranged
``GET``. To expand a glob, list the repository through the Hub's own API first — which is what
``vortex.datasets.load_dataset`` does — and then open each path it returns.

Hugging Face Datasets
=====================

``vortex.datasets.load_dataset`` builds on this to load Vortex files from the Hub as Hugging Face
``Datasets`` objects, expanding globs and pushing projections, filters and row limits into each
scan. See :doc:`../datasets`.
