# Upstreaming the Vortex loader to `huggingface/datasets`

This directory stages, **locally in the Vortex repo**, everything needed to add native
`.vortex` support to the Hugging Face [`datasets`](https://github.com/huggingface/datasets)
library, so that this works on any dataset repo containing Vortex files:

```python
import datasets

ds = datasets.load_dataset("my-org/my-dataset", streaming=True)
```

Nothing here is wired into any build — it is a staging area for a future PR against
`huggingface/datasets`. **Do not** open that PR until we are ready (see the checklist at
the bottom).

## What already works today (no upstream PR needed)

The `vortex-data` package itself (see `vortex-python/python/vortex/hf/`) already ships:

- `vortex.open("hf://datasets/org/name/data/train.vortex")` — lazy ranged reads from the
  Hub, with `HF_TOKEN`/token-cache auth.
- `vortex.hf.register_datasets()` — registers the same builder with a locally installed
  `datasets` at runtime:

  ```python
  import vortex.hf
  vortex.hf.register_datasets()

  import datasets
  ds = datasets.load_dataset("vortex", data_files={"train": "data/*.vortex"}, streaming=True)
  ```

The limitation of the runtime hook is that it only works after the user imports and calls
it, and the **Hub's dataset viewer will never call it**. Getting `.vortex` files to load
with plain `load_dataset("org/repo")` for everyone — and rendering in the Hub viewer —
requires the loader to ship *inside* `datasets`. That is what this directory stages.

## File map

| File in this directory | Destination in `huggingface/datasets` |
|---|---|
| `packaged_modules/vortex/__init__.py` | `src/datasets/packaged_modules/vortex/__init__.py` |
| `packaged_modules/vortex/vortex.py` | `src/datasets/packaged_modules/vortex/vortex.py` |
| `tests/test_vortex.py` | `tests/packaged_modules/test_vortex.py` |

The builder is a copy of `vortex-python/python/vortex/hf/builder.py` with two deliberate
differences (documented in its module docstring): `vortex` is imported lazily because it
is an optional dependency upstream, and a `token` config option is forwarded to
`vortex.hf.open` for `hf://` URIs (matching the `lance` builder's config surface).
**`vortex/hf/builder.py` is the source of truth** — if it changes, refresh the copy here.

Strip the `SPDX-License-Identifier` header lines when copying files upstream; `datasets`
source files carry no license headers (both projects are Apache-2.0).

## Supported `load_dataset` options

Every `VortexConfig` field is accepted as a `load_dataset(...)` keyword argument:

| Option | Type | Behavior |
|---|---|---|
| `columns` | `list[str]` | Column projection, pushed down to the Vortex scan. |
| `filters` | DNF tuples or `vortex.expr.Expr` | Predicate pushdown: `[("x", ">", 1)]` is an AND group, `[[...], [...]]` is an OR of AND groups; ops are `==`/`=`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`. Vortex evaluates predicates with zone maps and lazy segment reads, so in streaming mode non-matching segments are never downloaded. |
| `limit` | `int` | Maximum rows across all files (after filtering). Pushed down when no filter is set; otherwise the filter is pushed down and the limit is enforced in the builder (Vortex scans cannot combine both). |
| `indices` | `list[int]` | Explicit row indices, global across the split's files in listed order; deduplicated, rows returned in ascending order. Cannot be combined with `filters`/`limit`. |
| `batch_size` | `int` | Rows per generated Arrow batch. |
| `features` | `datasets.Features` | Explicit features instead of schema inference. |
| `token` | `str` | Hugging Face token for `hf://` data files (upstream copy only). |
| `on_bad_files` | `"error"` / `"warn"` / `"skip"` | Policy for files that fail to open as Vortex. |

The builder also implements `datasets.builder._CountableBuilderMixin`:
`_generate_num_examples` reads row counts from file footers without scanning data
(raises `NotImplementedError` when `filters`/`limit`/`indices` are set, since the
result count cannot be derived from footer metadata).

Two implementation notes worth knowing when reviewing:

- `datasets` requires the shard ids in keys yielded by `_generate_tables` to be dense
  (a file that yields no tables must not leave a gap), so the builder numbers *yielding*
  files, not files.
- `datasets` hashes non-default config kwargs with pickle for the cache fingerprint, and
  `vortex.Expr` is not picklable, so `VortexConfig.create_config_id` hashes the
  expression's stable string form instead.

## Step-by-step: making the upstream PR

1. **Fork and clone** `huggingface/datasets`, create a branch, and `pip install -e ".[dev]"`.

2. **Copy the module:**

   ```bash
   cp -r contrib/huggingface-datasets/packaged_modules/vortex \
       <datasets-repo>/src/datasets/packaged_modules/vortex
   cp contrib/huggingface-datasets/tests/test_vortex.py \
       <datasets-repo>/tests/packaged_modules/test_vortex.py
   ```

3. **Register the builder** in `src/datasets/packaged_modules/__init__.py`, exactly the
   way the `lance` module is registered there:

   ```python
   # with the other module imports at the top:
   from .vortex import vortex

   # in _PACKAGED_DATASETS_MODULES:
   "vortex": (vortex.__name__, _hash_python_lines(inspect.getsource(vortex).splitlines())),

   # in _EXTENSION_TO_MODULE:
   ".vortex": ("vortex", {}),
   ```

   Unlike Lance (a directory format with manifest/index sidecars that needs
   `_MODULE_TO_METADATA_FILE_NAMES` / `_MODULE_TO_METADATA_EXTENSIONS` entries), Vortex is
   a single-file format — the two entries above are all that is required. The
   `_MODULE_TO_EXTENSIONS` reverse mapping is derived automatically.

4. **Declare the optional dependency.** Find where `pylance` is declared in the
   `datasets` repo (`setup.py` test/dev extras at the time of writing) and mirror it with
   `vortex-data`, e.g. add `"vortex-data"` to `TESTS_REQUIRE` and, if the maintainers
   want a user-facing extra, `"vortex": ["vortex-data"]` in `EXTRAS_REQUIRE`. The import
   error raised by the builder already tells users to `pip install vortex-data`.

5. **Run the tests:**

   ```bash
   pytest tests/packaged_modules/test_vortex.py
   ```

   Also run the style hooks the repo requires (`make style` / `make quality` or
   pre-commit, per their `CONTRIBUTING.md`).

6. **Docs.** Two places, mirroring Lance:
   - `datasets` docs: add `.vortex` to the supported-formats tables in
     `docs/source/loading.mdx` (and the `Dataset formats` overview, if present).
   - `huggingface/hub-docs`: a `docs/hub/datasets-vortex.md` page modeled on
     [`datasets-lance.md`](https://github.com/huggingface/hub-docs/blob/main/docs/hub/datasets-lance.md),
     showing `vortex.open("hf://...")` and `load_dataset(..., streaming=True)`.

7. **Simplifications the reviewer may ask for:** the `Key` try/except compat shim in
   `vortex.py` exists so the same file also runs against older `datasets` releases via
   `vortex.hf.register_datasets()`; upstream, it can be collapsed to a plain
   `from datasets.builder import Key`.

## What only Hugging Face can do (raise in the PR / with HF contacts)

- **Dataset viewer rendering.** The Hub's viewer backend
  ([`huggingface/dataset-viewer`](https://github.com/huggingface/dataset-viewer)) is a
  separate deployment that pins its own dependencies. `.vortex` repos will not render on
  dataset pages until HF adds `vortex-data` to that deployment — merging the `datasets`
  PR is necessary but not sufficient. This is the collaboration step Lance/LanceDB went
  through.
- **Hub format recognition** (`format:vortex` filter, format badge on repo pages).

## Pre-PR readiness checklist

- [ ] `vortex-data` wheels published on PyPI for the Python versions and platforms the
      `datasets` CI matrix tests (manylinux x86_64/aarch64, macOS, Windows).
- [ ] `vortex.open` / `VortexFile.to_arrow` / `vortex.hf` API declared stable enough to
      be depended on by `datasets`.
- [ ] Builder copy here re-synced with `vortex-python/python/vortex/hf/builder.py`.
- [ ] An example public `.vortex` dataset repo on the Hub to point reviewers at.
- [ ] Tests in this directory pass against the `datasets` version we target (see below).

## Trying the staged module locally (without forking `datasets`)

The staged tests run against the builder *inside* a `datasets` checkout, but you can
smoke-test the identical behavior right now using the runtime registration:

```bash
cd vortex-python
uv run --all-packages pytest test/test_hf.py
```

which exercises the same builder via `vortex.hf.register_datasets()`, including a local
stand-in for the Hub's `resolve` endpoint (range requests, tokens) — no network needed.
