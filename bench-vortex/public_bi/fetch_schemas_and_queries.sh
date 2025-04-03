#!/bin/bash
# fetch all table definitions and queries from the public-bi benchmark
set -Eeuox pipefail

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}"

if [ -d ".git" ]; then
    git reset --hard  # restore deleted files if any
    git pull origin master
else
    git init
    git remote add origin "git@github.com:cwida/public_bi_benchmark.git"
    git config core.sparseCheckout true
    # checkout tables and queries under the benchmark folder only
    cat > .git/info/sparse-checkout << EOF
benchmark/*/tables/
benchmark/*/queries/
benchmark/*/data-urls.txt
EOF
    git fetch --depth 1 origin master
    git checkout origin/master
fi

# log HEAD
git rev-parse HEAD
