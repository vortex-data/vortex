#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

OS=$(uname -s)
function clear_caches() {
    sync
    if [ "$OS" = "Linux" ]; then
            echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    elif [ "$OS" = "Darwin" ]; then
            sudo purge
    fi
}


# Check if an argument is provided
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 [tpch|clickbench] <runs count>"
    exit 1
fi

if [ "$1" == "tpch" ] || [ "$1" == "clickbench" ]; then
    benchmark=$1
    echo "Running benchmark for $benchmark"
else
    echo "Invalid argument. Please use 'tpch' or 'clickbench'."
    exit 1
fi

if [ "$#" -eq 2 ]; then
    TRIES=$2
else
    TRIES=3
fi

if [ "$benchmark" = clickbench ]; then
    start_query=0
    end_query=42
else
    start_query=1
    end_query=22
fi

cargo build --bin $benchmark --package bench-vortex --profile samply --features mimalloc

echo "Generating data"
./target/samply/$benchmark -q $start_query -i 1 --formats vortex 1> /dev/null 2> /dev/null

echo "Running queries from ${start_query} to ${end_query} (inclusive)"



for query_num in $(seq $start_query $end_query); do
    clear_caches
    echo -n "Running query $query_num: ["
    for i in $(seq 1 $TRIES); do
        clear_caches

        RES=$(RUST_LOG=off ./target/samply/$benchmark  -i 1 --formats vortex --display-format gh-json -q $query_num --hide-progress-bar --hide-metrics | jq ".value / 1000000000")
        [[ $RES != "" ]] && \
            echo -n "$RES" || \
            echo -n "null"
        [[ "$i" != $TRIES ]] && echo -n ", "
    done
    echo "],"
done
