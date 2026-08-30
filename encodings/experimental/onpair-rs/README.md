# onpair-lib

Pure-Rust port of the training + encoding parts of
[`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp).

Scope is limited to what `vortex-onpair` actually consumes from
`vortex-onpair-sys`: `Column::compress` (BPE-style dictionary training plus
LSB-first bit-packed token encoding) and raw access to the resulting parts
(dictionary bytes/offsets, packed token stream, per-row boundaries). Decode,
LIKE, and EQ predicates are already pure Rust in `vortex-onpair` and reuse the
same `parts()` layout.

## Real-data training and parsing benchmarks

The benchmark consumes byte-capped corpus files rather than downloading data
inside a timed process. Install `datasets`, then materialize one or more 32 MiB
samples:

```bash
python -m pip install datasets
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py fineweb
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py clickbench
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py stack-v3
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py msmarco-queries
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py msmarco-urls
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py amazon-book-titles
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py dbpedia-abstracts
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py apache-access
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py paper-book-reviews
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py paper-news-headlines
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py paper-tweets
```

The Hugging Face sources are pinned to repository revisions. ClickBench uses
the official partitioned Parquet dataset. The four OnPair paper datasets use
the exact URLs and parsing rules from the paper artifact at revision
`ef3360530e9e963dedc3b59280b5bc2014ce7416`.

TPC-H `lineitem.l_comment` uses the repository's pinned `tpchgen-rs`. Generate
SF1 Parquet data, then materialize the string sample:

```bash
cargo run -p vortex-bench --bin data-gen -- tpch
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py tpch-l-comment
```

To prepare another Parquet corpus, specify its string column explicitly:

```bash
python encodings/experimental/onpair-rs/scripts/prepare_corpora.py parquet \
  --source /tmp/onpair-short-profile.parquet --column URL --name onpair-profile
```

Each `.onpair` file has exactly 32 MiB of string payload by default, plus row
framing; its JSON sidecar records provenance and a SHA-256 checksum. Only the
final row may be truncated to reach the exact byte target. Run only the
dictionary-training and parsing benchmarks for each sample with:

```bash
ONPAIR_BENCH_CORPUS=/tmp/onpair-corpora/fineweb-32mib.onpair \
  cargo bench -p vortex-onpair-rs --bench clickbench
```

Set `ONPAIR_BITS=16` to exercise the full 65,536-token dictionary. The
single-threaded upstream C++ baseline consumes the same corpus and uses the
same training configuration. Build it against the pinned `onpair_cpp` source
with native optimization and link-time optimization:

```bash
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG \
  -I "$ONPAIR_CPP_SRC/include" \
  encodings/experimental/onpair-rs/benches/onpair_cpp_baseline.cpp \
  "$ONPAIR_CPP_SRC/src/onpair/core/dictionary_view.cpp" \
  "$ONPAIR_CPP_SRC/src/onpair/encoding/training/trainer.cpp" \
  "$ONPAIR_CPP_SRC/src/onpair/encoding/parsing/parser.cpp" \
  -o /tmp/onpair_cpp_baseline

ONPAIR_BITS=16 ONPAIR_SAMPLE_FRACTION=0.15 ONPAIR_REPORT_ITERATIONS=7 \
  /tmp/onpair_cpp_baseline /tmp/onpair-corpora/fineweb-32mib.onpair
```

`ONPAIR_SAMPLE_FRACTION` defaults to `0.5`; the paper benchmark wrapper uses
`0.15`.
