#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ONPAIR_CPP_DIR BOOST_DIR OUTPUT_DIR" >&2
  exit 2
fi

cpp_dir=$1
boost_dir=$2
output_dir=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_dir"

sources=(
  "$script_dir/onpair_cpp_bench.cpp"
  "$cpp_dir/src/onpair/column/column.cpp"
  "$cpp_dir/src/onpair/core/dictionary_view.cpp"
  "$cpp_dir/src/onpair/encoding/parsing/parser.cpp"
  "$cpp_dir/src/onpair/encoding/training/trainer.cpp"
)
common=(-std=c++20 -O3 -DNDEBUG -march=native -I"$cpp_dir/include" -I"$boost_dir")

g++ "${common[@]}" -flto "${sources[@]}" -o "$output_dir/onpair-gcc"
clang++ "${common[@]}" -flto=thin "${sources[@]}" -o "$output_dir/onpair-clang"
