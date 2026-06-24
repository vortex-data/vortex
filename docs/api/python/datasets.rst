Hugging Face Datasets
=====================

Vortex files can be loaded directly as Hugging Face ``datasets`` objects. Install the optional
dependencies with:

.. code-block:: bash

    pip install vortex-data[hf]

``vortex.datasets.load_dataset`` accepts a local path, a glob, or a Hugging Face Hub dataset
repository id. Unlike ``datasets.load_dataset``, it defaults to ``streaming=True`` and returns a
streaming dataset that keeps Vortex in charge of reading. Column selection, Vortex filter
expressions, and row limits are pushed down into each Vortex scan before examples are yielded to
Hugging Face transforms. Pass ``streaming=False`` to eagerly materialize an in-memory dataset.

.. code-block:: python

    import vortex as vx

    # Stream a local directory of Vortex files as a Hugging Face IterableDataset.
    ds = vx.datasets.load_dataset("path/to/dir", split="train")

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
