# Fuzzing Coverage & Corpus Management

This document explains how to view coverage reports, minimize the corpus, and visualize fuzzing effectiveness.

## Table of Contents

- [Coverage Reports](#coverage-reports)
- [Corpus Minimization](#corpus-minimization)
- [Coverage Visualizers](#coverage-visualizers)
- [CI Integration](#ci-integration)

---

## Coverage Reports

Coverage tracking measures which code paths the fuzzer exercises, helping identify blind spots.

### Generate Coverage Data

```bash
# Run fuzzer with coverage instrumentation
cargo +nightly fuzz coverage array_ops

# This creates: fuzz/coverage/array_ops/coverage.profdata
```

### Generate HTML Report

```bash
# Find the instrumented binary
FUZZ_BIN=$(find target -name "array_ops" -path "*/coverage/*" -type f | head -1)

# Generate HTML report
llvm-cov show "$FUZZ_BIN" \
    -instr-profile=fuzz/coverage/array_ops/coverage.profdata \
    -format=html \
    -output-dir=coverage-html \
    -Xdemangler=rustfilt \
    -ignore-filename-regex='/.cargo/|/rustc/'

# Open in browser
open coverage-html/index.html      # macOS
xdg-open coverage-html/index.html  # Linux
```

### Generate Summary Report

```bash
# Text summary to terminal
llvm-cov report "$FUZZ_BIN" \
    -instr-profile=fuzz/coverage/array_ops/coverage.profdata \
    -ignore-filename-regex='/.cargo/|/rustc/'
```

Output:
```
Filename                                    Regions    Miss   Cover   Lines    Miss   Cover
--------------------------------------------------------------------------------------------
vortex-array/src/compute/slice.rs               45       2   95.56%     120       6   95.00%
vortex-array/src/compute/take.rs                60      22   63.33%     150      55   63.33%
vortex-array/src/compute/filter.rs              40       5   87.50%     100      11   89.00%
--------------------------------------------------------------------------------------------
TOTAL                                          145      29   80.00%     370      72   80.54%
```

### Export to LCOV Format

For use with other tools (Codecov, VS Code, etc.):

```bash
llvm-cov export "$FUZZ_BIN" \
    -instr-profile=fuzz/coverage/array_ops/coverage.profdata \
    -format=lcov \
    -ignore-filename-regex='/.cargo/|/rustc/' \
    > coverage.lcov
```

---

## Corpus Minimization

Over time, the corpus accumulates redundant inputs that cover the same code paths. `cmin` removes them.

### Why Minimize?

| Metric | Before cmin | After cmin |
|--------|-------------|------------|
| Inputs | 5,000 | 300 |
| Size | 250 MB | 15 MB |
| Edge coverage | 15,000 | 15,000 (same!) |

Benefits:
- Faster fuzzer startup
- Less storage
- More efficient fuzzing

### Run Corpus Minimization

```bash
# Ensure corpus is in place
ls fuzz/corpus/array_ops/

# Measure before
echo "Before: $(find fuzz/corpus/array_ops -type f | wc -l) inputs"
du -sh fuzz/corpus/array_ops

# Minimize
cargo +nightly fuzz cmin array_ops

# Measure after
echo "After: $(find fuzz/corpus/array_ops -type f | wc -l) inputs"
du -sh fuzz/corpus/array_ops
```

### How cmin Works

1. Runs each corpus input through instrumented binary
2. Records which code edges each input covers
3. Uses greedy set-cover to find minimal set covering all edges
4. Removes redundant inputs

```
Input A: covers edges {1, 2, 3, 5}
Input B: covers edges {1, 2, 3}      <- subset of A, removed
Input C: covers edges {4, 6, 7}
Input D: covers edges {4, 6}         <- subset of C, removed

Result: A, C kept; B, D removed
```

### Minimize Remote Corpus

If corpus is stored in S3/R2:

```bash
# Download
aws s3 cp s3://vortex-fuzz-corpus/array_ops_corpus.tar.zst .
mkdir -p fuzz/corpus/array_ops
tar -xf array_ops_corpus.tar.zst -C fuzz/corpus/array_ops

# Minimize
cargo +nightly fuzz cmin array_ops

# Re-upload
tar -cf - -C fuzz/corpus array_ops | zstd > array_ops_corpus.tar.zst
aws s3 cp array_ops_corpus.tar.zst s3://vortex-fuzz-corpus/
```

---

## Coverage Visualizers

### 1. HTML Report (Built-in)

The `llvm-cov show -format=html` command produces a browsable report:

- File tree with coverage percentages
- Click files to see line-by-line coverage
- Hit counts for each line
- Branch coverage indicators

### 2. VS Code Coverage Gutters

Install the "Coverage Gutters" VS Code extension, then:

```bash
# Generate lcov format
llvm-cov export "$FUZZ_BIN" \
    -instr-profile=fuzz/coverage/array_ops/coverage.profdata \
    -format=lcov > lcov.info
```

VS Code shows coverage inline:
- Green gutter = line covered
- Red gutter = line not covered
- Click "Watch" in status bar to enable

### 3. Terminal Summary Script

Create `scripts/coverage-summary.sh`:

```bash
#!/bin/bash
# Quick terminal coverage visualization

TARGET="${1:-array_ops}"

cargo +nightly fuzz coverage "$TARGET" 2>/dev/null

FUZZ_BIN=$(find target -name "$TARGET" -path "*/coverage/*" -type f | head -1)

llvm-cov report "$FUZZ_BIN" \
    -instr-profile="fuzz/coverage/$TARGET/coverage.profdata" \
    -ignore-filename-regex='/.cargo/|/rustc/' 2>/dev/null | \
awk '
/^---/ { next }
/TOTAL/ {
    printf "\n\033[1mTOTAL: %.1f%%\033[0m\n", $10
    next
}
/vortex/ {
    pct = $10 + 0
    bar = ""
    for (i = 0; i < pct/5; i++) bar = bar "#"
    for (i = pct/5; i < 20; i++) bar = bar "-"

    if (pct >= 80) color = "\033[32m"
    else if (pct >= 50) color = "\033[33m"
    else color = "\033[31m"

    printf "%-50s %s[%s] %5.1f%%\033[0m\n", $1, color, bar, pct
}'
```

Usage:
```bash
./scripts/coverage-summary.sh array_ops
```

Output:
```
vortex-array/src/compute/slice.rs                  [###################-] 95.0%
vortex-array/src/compute/take.rs                   [############--------] 63.3%
vortex-array/src/compute/filter.rs                 [#################---] 89.0%
vortex-array/src/arrays/dict.rs                    [#########-----------] 45.3%

TOTAL: 73.2%
```

### 4. Codecov Integration

For tracking coverage over time with graphs and PR comments:

```yaml
# In CI workflow
- name: Upload to Codecov
  uses: codecov/codecov-action@v4
  with:
    files: coverage.lcov
    flags: fuzzing
```

### 5. grcov (Mozilla)

Alternative HTML generator with cleaner styling:

```bash
cargo install grcov

grcov . \
    --binary-path ./target/coverage/ \
    --source-dir . \
    --output-types html \
    --branch \
    --ignore-not-existing \
    --output-path ./coverage-grcov

open coverage-grcov/index.html
```

---

## CI Integration

### Weekly Corpus Minimization

```yaml
name: Minimize Fuzz Corpus
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly Sunday
  workflow_dispatch:

jobs:
  minimize:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [array_ops, file_io, compress_roundtrip]
    steps:
      - uses: actions/checkout@v4

      - name: Setup
        run: |
          rustup install nightly
          cargo install cargo-fuzz

      - name: Download corpus
        run: |
          aws s3 cp s3://vortex-fuzz-corpus/${{ matrix.target }}_corpus.tar.zst .
          mkdir -p fuzz/corpus/${{ matrix.target }}
          tar -xf *.tar.zst -C fuzz/corpus/${{ matrix.target }}

      - name: Measure before
        id: before
        run: |
          echo "count=$(find fuzz/corpus/${{ matrix.target }} -type f | wc -l)" >> $GITHUB_OUTPUT

      - name: Minimize
        run: cargo +nightly fuzz cmin ${{ matrix.target }}

      - name: Measure after
        id: after
        run: |
          echo "count=$(find fuzz/corpus/${{ matrix.target }} -type f | wc -l)" >> $GITHUB_OUTPUT

      - name: Upload minimized
        run: |
          tar -cf - -C fuzz/corpus ${{ matrix.target }} | zstd > corpus.tar.zst
          aws s3 cp corpus.tar.zst s3://vortex-fuzz-corpus/${{ matrix.target }}_corpus.tar.zst

      - name: Summary
        run: |
          echo "Minimized ${{ matrix.target }}: ${{ steps.before.outputs.count }} -> ${{ steps.after.outputs.count }} inputs"
```

### Coverage Reporting

```yaml
name: Fuzz Coverage Report
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday
  workflow_dispatch:

jobs:
  coverage:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [array_ops, file_io, compress_roundtrip]
    steps:
      - uses: actions/checkout@v4

      - name: Setup
        run: |
          rustup install nightly
          cargo install cargo-fuzz

      - name: Download corpus
        run: |
          aws s3 cp s3://vortex-fuzz-corpus/${{ matrix.target }}_corpus.tar.zst .
          mkdir -p fuzz/corpus/${{ matrix.target }}
          tar -xf *.tar.zst -C fuzz/corpus/${{ matrix.target }}

      - name: Generate coverage
        run: cargo +nightly fuzz coverage ${{ matrix.target }}

      - name: Create HTML report
        run: |
          FUZZ_BIN=$(find target -name "${{ matrix.target }}" -path "*/coverage/*" -type f | head -1)
          llvm-cov show "$FUZZ_BIN" \
              -instr-profile=fuzz/coverage/${{ matrix.target }}/coverage.profdata \
              -format=html \
              -output-dir=coverage-html \
              -ignore-filename-regex='/.cargo/|/rustc/'

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.target }}
          path: coverage-html/
          retention-days: 30

      - name: Export LCOV
        run: |
          FUZZ_BIN=$(find target -name "${{ matrix.target }}" -path "*/coverage/*" -type f | head -1)
          llvm-cov export "$FUZZ_BIN" \
              -instr-profile=fuzz/coverage/${{ matrix.target }}/coverage.profdata \
              -format=lcov > coverage.lcov

      - name: Upload to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage.lcov
          flags: ${{ matrix.target }}
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Generate coverage | `cargo +nightly fuzz coverage <target>` |
| HTML report | `llvm-cov show ... -format=html -output-dir=dir` |
| Text summary | `llvm-cov report ...` |
| LCOV export | `llvm-cov export ... -format=lcov` |
| Minimize corpus | `cargo +nightly fuzz cmin <target>` |

## Interpreting Coverage

| Coverage | Status | Action |
|----------|--------|--------|
| > 80% | Good | Maintain |
| 50-80% | Moderate | Add seeds for uncovered paths |
| < 50% | Poor | Investigate, add dictionary/seeds |

Low coverage areas indicate:
- Code paths the fuzzer hasn't found
- Potential need for targeted seeds
- Possible dead code
