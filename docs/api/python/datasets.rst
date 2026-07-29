Hugging Face Datasets
=====================

Vortex files can be loaded directly as Hugging Face ``datasets`` objects. Install the optional
dependencies with:

.. code-block:: bash

    pip install vortex-data[hf]

``vortex.datasets.load_dataset`` accepts a local path or glob, a URL readable by Vortex
(``https://``, ``s3://``, ``gs://``, ``az://``), a Hugging Face Hub dataset repository id, or an
``hf://`` URI of the form ``hf://datasets/<org>/<name>[@<revision>][/<path-or-glob>]`` (revisions
containing ``/`` must be percent-encoded, e.g. ``@refs%2Fconvert%2Fparquet``).
Unlike ``datasets.load_dataset``, it defaults to ``streaming=True`` and returns a streaming
dataset that keeps Vortex in charge of reading. Column selection, Vortex filter expressions, and
row limits are pushed down into each Vortex scan before examples are yielded to Hugging Face
transforms.

Hub repositories are streamed in place: files are read with HTTP range requests, so only the
projected columns and matching rows are ever transferred. Private and gated repositories
authenticate with the ``token`` argument or the locally saved login. Files are downloaded (with
the usual Hub caching) only when ``streaming=False`` or ``local_files_only=True``.

.. code-block:: python

    import vortex as vx

    # Stream a local directory of Vortex files as a Hugging Face IterableDataset.
    ds = vx.datasets.load_dataset("path/to/dir", split="train")

    # Stream a Hub dataset directly, without downloading it.
    ds = vx.datasets.load_dataset("username/dataset", split="train")

    # hf:// URIs select a revision and a path or glob within the repository.
    ds = vx.datasets.load_dataset("hf://datasets/username/dataset@main/data/*.vortex", split="train")

    # Column projection and Vortex filter expressions are pushed into the scan.
    ds = ds.select_columns(["text", "label"]).filter(vx.expr.column("label") == 1)

    for example in ds.take(100):
        ...

    # Materialize an in-memory dataset instead, optionally selecting splits via data_files.
    splits = vx.datasets.load_dataset(
        "username/dataset",
        data_files={"train": "train-*.vortex", "validation": "validation.vortex"},
        streaming=False,
    )

When loading multiple splits, pass a ``data_files`` mapping from split name to file patterns; a
list of splits without such a mapping cannot be resolved and raises an error.
