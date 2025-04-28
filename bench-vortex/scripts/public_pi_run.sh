#!/bin/bash

# List files in the current directory
files=$(ls ../public_bi/benchmark)

# Loop through each file
for file in $files; do
    # Check if it's a regular file
      echo "Running public BI: $file"
      
      file_lowercase=$(echo "$file" | tr '[:upper:]' '[:lower:]')

      cargo run --profile bench --bin public_bi -- --targets=datafusion:vortex,duckdb:vortex -d $file_lowercase -i1

      echo ""
done
