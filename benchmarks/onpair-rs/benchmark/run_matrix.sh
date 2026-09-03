#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CORPUS_DIR RUST_BIN ORIGINAL_RUST_BIN GCC_BIN CLANG_BIN" >&2
  exit 2
fi

corpus_dir=$1
rust_bin=$2
original_rust_bin=$3
gcc_bin=$4
clang_bin=$5
cpu=${ONPAIR_CPU:-0}

corpora=(
  amazon-book-titles-32mib
  apache-access-32mib
  clickbench-32mib
  dbpedia-abstracts-32mib
  fineweb-32mib
  msmarco-queries-32mib
  msmarco-urls-32mib
  onpair-titles-32mib
  paper-book-reviews-32mib
  paper-news-headlines-32mib
  paper-tweets-32mib
  stack-v3-32mib
  tpch-l-comment-32mib
  fineweb-128mib
  fineweb-shard1-128mib
)

for corpus in "${corpora[@]}"; do
  file="$corpus_dir/$corpus.onpair"
  if [[ -n ${ONPAIR_WARMUPS-} ]]; then
    warmups=$ONPAIR_WARMUPS
  elif [[ $corpus == *128mib ]]; then
    warmups=1
  else
    warmups=2
  fi
  if [[ -n ${ONPAIR_ITERATIONS-} ]]; then
    iterations=$ONPAIR_ITERATIONS
  elif [[ $corpus == *128mib ]]; then
    iterations=3
  else
    iterations=5
  fi
  for bits in 12 16; do
    for implementation in rust gcc clang; do
      case "$implementation" in
        rust) binary=$rust_bin ;;
        gcc) binary=$gcc_bin ;;
        clang) binary=$clang_bin ;;
      esac
      echo "BEGIN corpus=$corpus implementation=$implementation bits=$bits"
      ONPAIR_BITS=$bits ONPAIR_WARMUPS=$warmups ONPAIR_ITERATIONS=$iterations \
        taskset -c "$cpu" "$binary" "$file"
    done
  done
  echo "BEGIN corpus=$corpus implementation=rust_original bits=16"
  ONPAIR_WARMUPS=$warmups ONPAIR_ITERATIONS=$iterations \
    taskset -c "$cpu" "$original_rust_bin" "$file"
done
