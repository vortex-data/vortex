#!/bin/bash

set -Eeu -o pipefail -x

# List files in the current directory
readonly dir=$(dirname ${BASH_SOURCE[0]})
readonly files=$(ls $dir/../public_bi/benchmark)

for file in $files; do
    echo "Running public BI: $file"
      
    file_lowercase=$(echo "$file" | tr '[:upper:]' '[:lower:]')

    cargo run --profile bench --bin public_bi -- --targets=datafusion:vortex,duckdb:vortex -d $file_lowercase -i1

    echo ""
done
