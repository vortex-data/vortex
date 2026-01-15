# Fuzzer Crash Reporting Scripts

Scripts for automated fuzzer crash deduplication and issue reporting.

## Deduplication Chain

When a fuzzer crash is detected, the following checks are run in order. If **any** check matches, the crash is considered a duplicate:

| Order | Script | Confidence | Description |
|-------|--------|------------|-------------|
| 1 | `check-seed-hash.sh` | Exact | Same crash input file (100% duplicate) |
| 2 | `check-panic-location.sh` | High | Same file:line crash site |
| 3 | `check-stack-trace.sh` | High | Same top 5 stack frames |
| 4 | `check-error-pattern.sh` | Medium | Same normalized error message |
| 5 | Claude Code Action | Medium | Semantic similarity to existing issues |

## Scripts

### `extract-crash-info.sh`

Parses fuzzer output logs and crash artifacts to extract:
- Panic location (file:line)
- Error message and variant
- Stack trace frames
- Computes hashes for deduplication

```bash
./extract-crash-info.sh <log_file> <crash_file> [output_json]
```

Output JSON:
```json
{
  "panic_location": "vortex-array/src/compute/slice.rs:142",
  "panic_message": "index out of bounds",
  "error_variant": "ScalarMismatch",
  "stack_frames": ["frame1", "frame2", ...],
  "stack_trace_hash": "abc123...",
  "message_hash": "def456...",
  "crash_type": "crash",
  "seed_hash": "789xyz..."
}
```

### `check-duplicate.sh`

Chains all deduplication checks together.

```bash
./check-duplicate.sh <crash_info.json> [issues.json]
```

Output JSON:
```json
{
  "duplicate": true,
  "check": "seed_hash",
  "confidence": "exact",
  "issue_number": 123,
  "reason": "Exact seed hash match"
}
```

### `render-template.sh`

Substitutes `{{VAR}}` placeholders in templates with environment variables.

```bash
export FUZZ_TARGET="file_io"
export CRASH_FILE="crash-abc123"
./render-template.sh fuzzer-crash.template.md output.md
```

## Templates

### `fuzzer-crash.template.md`

Template for new issue creation. Includes:
- Crash summary table
- Deduplication hashes (collapsible)
- Error details
- Claude analysis section
- Reproduction instructions

### `fuzzer-crash-comment.template.md`

Template for comments on existing issues. Includes:
- Match details (why it was marked as duplicate)
- New crash information
- Claude analysis

## Workflow Integration

These scripts are used by `.github/workflows/report-fuzz-crash.yml`:

```
fuzz.yml (detects crash)
    │
    ▼
report-fuzz-crash.yml
    │
    ├─► extract-crash-info.sh (parse logs)
    ├─► check-duplicate.sh (scripted checks)
    ├─► Claude dedup check (if no match)
    ├─► Claude analysis (root cause)
    └─► render-template.sh + gh issue create/comment
```

## Adding New Checks

To add a new deduplication check:

1. Create `check-<name>.sh` that outputs JSON with `match`, `confidence`, `issue_number`, `reason`
2. Add it to the chain in `check-duplicate.sh`
3. Update this README

## Testing Locally

```bash
# Extract crash info from a log
./extract-crash-info.sh fuzz_output.log crash-abc123 crash_info.json

# Check for duplicates (needs fuzzer_issues.json from gh issue list)
gh issue list --label fuzzer --state open --json number,title,body,url > fuzzer_issues.json
./check-duplicate.sh crash_info.json fuzzer_issues.json

# Render a template
export FUZZ_TARGET="file_io"
# ... set other vars
./render-template.sh fuzzer-crash.template.md issue_body.md
```
