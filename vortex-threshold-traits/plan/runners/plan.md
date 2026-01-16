# CI Runners & Infrastructure

## Overview

The CI infrastructure runs benchmarks across multiple CPU architectures, collects results, aggregates them, and optionally updates threshold files or comments on PRs.

## Current Implementation

### Done
- [x] GitHub Actions workflow template (`.github/workflows/isa-thresholds.yml`)
- [x] Runner matrix definition for multiple architectures
- [x] Artifact upload/download for result collection
- [x] Basic aggregation job that merges JSON files

### Not Done
- [ ] Actual runner labels configured in GitHub
- [ ] Workflow tested end-to-end
- [ ] PR comment with threshold changes
- [ ] Status check for threshold regressions
- [ ] Scheduled runs for trend tracking
- [ ] Self-hosted runner setup documentation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions Workflow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Trigger: push to main, PR, schedule                                    │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │Intel Sapphire│  │ AMD Genoa   │  │ Graviton 3  │  │ Graviton 2  │    │
│  │   Runner    │  │   Runner    │  │   Runner    │  │   Runner    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│         v                v                v                v            │
│     results-         results-         results-         results-         │
│     intel.json       amd.json         arm3.json        arm2.json        │
│         │                │                │                │            │
│         └────────────────┴────────────────┴────────────────┘            │
│                                    │                                    │
│                                    v                                    │
│                          ┌─────────────────┐                            │
│                          │   Aggregator    │                            │
│                          │   (ubuntu-latest)│                           │
│                          └────────┬────────┘                            │
│                                   │                                     │
│                    ┌──────────────┼──────────────┐                      │
│                    v              v              v                      │
│              thresholds.rs   PR Comment    Status Check                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Runner Matrix

| Label | Architecture | CPU | Features | Provider |
|-------|-------------|-----|----------|----------|
| `intel-sapphire` | x86_64 | Intel Sapphire Rapids | AVX-512, AMX | Self-hosted |
| `intel-icelake` | x86_64 | Intel Ice Lake | AVX-512 | Self-hosted |
| `amd-genoa` | x86_64 | AMD EPYC Genoa (Zen 4) | AVX-512 | Self-hosted |
| `amd-milan` | x86_64 | AMD EPYC Milan (Zen 3) | AVX2 | Self-hosted |
| `graviton3` | aarch64 | AWS Graviton 3 | NEON, SVE | AWS CodeBuild / Self-hosted |
| `graviton2` | aarch64 | AWS Graviton 2 | NEON | AWS CodeBuild / Self-hosted |

## Workflow Configuration

### Current Template

```yaml
name: ISA Threshold Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  benchmark:
    strategy:
      fail-fast: false
      matrix:
        include:
          - runner: intel-sapphire
            cpu_class: IntelSapphire
          - runner: amd-genoa
            cpu_class: AmdGenoa
          - runner: graviton3
            cpu_class: Graviton3

    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build runner
        run: cargo build -p vortex-threshold-runner --release

      - name: Run benchmarks
        run: |
          ./target/release/threshold-runner \
            --output results-${{ matrix.cpu_class }}.json \
            --commit ${{ github.sha }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: results-${{ matrix.cpu_class }}
          path: results-${{ matrix.cpu_class }}.json

  aggregate:
    needs: benchmark
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download all results
        uses: actions/download-artifact@v4
        with:
          pattern: results-*
          merge-multiple: true

      - name: Build aggregator
        run: cargo build -p vortex-threshold-aggregator --release

      - name: Aggregate results
        run: |
          ./target/release/threshold-aggregator \
            --input results-*.json \
            --output thresholds/generated.rs

      - name: Upload thresholds
        uses: actions/upload-artifact@v4
        with:
          name: thresholds
          path: thresholds/generated.rs
```

## Planned Changes

### 1. PR Comments

Post a comment on PRs showing threshold changes:

```yaml
- name: Post PR comment
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const diff = fs.readFileSync('threshold-diff.md', 'utf8');
      github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body: diff
      });
```

Example comment:
```markdown
## ISA Threshold Changes

| Algorithm | CPU | Before | After | Change |
|-----------|-----|--------|-------|--------|
| popcount | IntelSapphire | 256 | 512 | +100% :warning: |
| popcount | AmdGenoa | 512 | 512 | 0% |
| sum | Graviton3 | 128 | 96 | -25% :white_check_mark: |

:warning: Threshold increased by >50% for `popcount` on Intel - please verify this is expected.
```

### 2. Status Check

Fail the check if thresholds regress significantly:

```yaml
- name: Check for regressions
  run: |
    ./target/release/threshold-aggregator \
      --check-regression \
      --max-increase-percent 50 \
      --base thresholds/current.rs \
      --head thresholds/generated.rs
```

### 3. Scheduled Trend Tracking

```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    # ... run benchmarks ...

  store-history:
    needs: benchmark
    runs-on: ubuntu-latest
    steps:
      - name: Store to SQLite
        run: |
          ./target/release/threshold-runner \
            --store-to-db benchmarks.db \
            --input results-*.json

      - name: Upload database
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-history
          path: benchmarks.db
          retention-days: 90
```

### 4. Self-Hosted Runner Setup

Documentation for setting up self-hosted runners:

```bash
# On Intel Sapphire Rapids machine
./actions-runner/config.sh \
  --url https://github.com/spiraldb/vortex \
  --token <TOKEN> \
  --labels intel-sapphire,x86_64,avx512

# On AWS Graviton 3 instance
./actions-runner/config.sh \
  --url https://github.com/spiraldb/vortex \
  --token <TOKEN> \
  --labels graviton3,aarch64,neon,sve
```

## Open Questions

1. **Runner availability**: Do we have access to all these CPU types?
2. **Cost**: Self-hosted vs. cloud instances for benchmarking?
3. **Isolation**: How to ensure consistent benchmark results on shared runners?
4. **Frequency**: How often to run benchmarks? Every PR? Only main? Scheduled?
5. **Fallback**: What if a runner is unavailable? Skip or fail?

## Files

- `.github/workflows/isa-thresholds.yml` - Main workflow
- `vortex-threshold-aggregator/src/main.rs` - Aggregation and diff generation

## Next Steps

1. Set up at least one self-hosted runner for testing
2. Test workflow end-to-end on a single architecture
3. Add PR comment generation
4. Add regression check status
5. Document runner setup process
