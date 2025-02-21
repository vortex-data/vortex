#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if ! [ -x $(which gtime) ]
then
    echo "It appears you're missing gtime, its available on brew as 'gnu-time'.";
    exit 1;
fi

export RUST_LOG="OFF"
OUTPUT_FILE="output.txt"
echo "Writing results to $OUTPUT_FILE"

# Clear output file
> $OUTPUT_FILE

echo "Building binaries with mimalloc"
cargo build --bin tpch --bin clickbench --release --features mimalloc

for i in "tpch 1 22" "clickbench 0 42"; do
    set -- $i
    benchmark=$1
    range_start=$2
    range_end=$3

    for q in $(seq $range_start $range_end); do
        echo "running $benchmark q $q";

        gtime -f %M target/release/$benchmark -q $q -i 2 --formats vortex 2>> $OUTPUT_FILE
    done
done
