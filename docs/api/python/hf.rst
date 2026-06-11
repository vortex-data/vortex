================
Hugging Face Hub
================

Vortex files hosted on the `Hugging Face Hub <https://huggingface.co>`_ can be read
directly with ``hf://`` URLs, following the same convention as ``huggingface_hub``'s
``HfFileSystem``::

    hf://datasets/{namespace}/{name}[@{revision}]/{path/in/repo}

URLs are translated into ranged HTTP reads against the Hub's ``resolve`` endpoint, so
lazy scans (projection, predicate pushdown, row indices) download only the bytes they
need rather than whole files. The ``HF_ENDPOINT`` environment variable is honored, and
tokens for gated or private repositories are resolved from ``HF_TOKEN`` or the
``huggingface_hub`` token cache.

.. autosummary::
   :nosignatures:

   ~vortex.hf.open
   ~vortex.hf.HFLocation
   ~vortex.hf.resolve_url
   ~vortex.hf.http_store
   ~vortex.hf.store_and_path
   ~vortex.hf.token
   ~vortex.hf.endpoint

Example
-------

Convert an existing Parquet shard to Vortex, publish it to a dataset repository, and
read it back lazily (publishing requires the ``huggingface_hub`` package and an access
token):

.. code-block:: python

   import pyarrow.parquet as pq
   import vortex as vx

   # Convert a Parquet shard to Vortex.
   vx.io.write(pq.read_table("train-00000.parquet"), "train-00000.vortex")

   # Publish it to the Hub.
   from huggingface_hub import HfApi

   HfApi().upload_file(
       path_or_fileobj="train-00000.vortex",
       path_in_repo="data/train-00000.vortex",
       repo_id="my-org/my-dataset",
       repo_type="dataset",
   )

   # Anyone can now open the file lazily, without downloading all of it.
   vxf = vx.open("hf://datasets/my-org/my-dataset/data/train-00000.vortex")
   scores = vxf.scan(["score"]).read_all()

.. raw:: html

   <hr>

.. automodule:: vortex.hf
    :members:
    :imported-members:
