# Automated Fuzzer Crash Analysis with Claude Code

This directory contains workflows for automated fuzzer crash detection, analysis, and issue creation using the Claude Code bot.

## Overview

The fuzzing infrastructure automatically detects crashes and uses Claude to analyze them and create/update GitHub issues with duplicate detection.

## How It Works

### 1. Crash Detection and Analysis (fuzz.yml)

The fuzzing workflow runs every 4 hours and:

1. **Runs Fuzzing Targets** - Executes `file_io` and `array_ops` targets for 2 hours each
2. **Detects Crashes** - Checks for crash files in `fuzz/artifacts`
3. **Archives Artifacts** - Saves crash files and fuzzer logs
4. **Triggers Claude Analysis** - Spawns a separate job to analyze the crash

**Fuzzing Targets:**
- `file_io` - Tests file I/O operations
- `array_ops` - Tests array operations

### 2. Claude Automated Analysis (report-*-fuzz-failures jobs)

When crashes are detected, Claude automatically:

1. **Reads Fuzzer Output** - Parses `fuzz_output.log` to extract:
   - Stack trace (frames with `#0`, `#1`, etc.)
   - Error message (panic message or ERROR)
   - Crash location (top user code frame, excluding std/core/libfuzzer)
   - Debug output (from `std::fmt::Debug` section)

2. **Analyzes Root Cause** - Reads source code at crash location to understand the issue

3. **Checks for Duplicates** - Searches existing issues labeled `fuzzer`:
   - Compares crash location (file + function, ignoring line numbers)
   - Compares error patterns (normalizing values, e.g., "index 5" = "index 12")
   - Reads source code to verify same root cause
   - Determines: EXACT DUPLICATE, SIMILAR, or NEW BUG

4. **Takes Action Based on Classification**:

   **EXACT DUPLICATE (high confidence):**
   - Updates or creates a tracking comment with occurrence count
   - Comment format: `<!-- occurrences: N -->` followed by latest occurrence details
   - Uses GitHub API to find and edit existing tracking comments

   **SIMILAR (medium confidence):**
   - Adds a new comment explaining the similarity
   - Includes confidence level and reasoning
   - Notes key differences if any

   **NEW BUG (not a duplicate):**
   - Creates a new issue with label `bug,fuzzer`
   - Includes detailed analysis, stack trace, root cause, reproduction steps

## Issue Format

Claude creates issues with this structure:

```markdown
## Fuzzing Crash Report

### Analysis

**Crash Location**: `file.rs:function_name`

**Error Message**:
```
[error message]
```

**Stack Trace**:
```
[top 5-7 frames in code block]
```

**Root Cause**: [Claude's analysis of the underlying issue]

<details>
<summary>Debug Output</summary>

```
[Complete "Output of std::fmt::Debug:" section from fuzzer log]
```
</details>

### Summary

- **Target**: `file_io` or `array_ops`
- **Crash File**: `crash-<hash>`
- **Branch**: [branch name]
- **Commit**: [commit SHA]
- **Crash Artifact**: [direct download link]

### Reproduction

1. Download the crash artifact:
   - **Direct download**: [artifact URL]
   - Extract the zip file

2. Reproduce locally:
```bash
# The artifact contains <target>/<crash-file>
cargo +nightly fuzz run --sanitizer=none <target> <target>/<crash-file>
```

3. Get full backtrace:
```bash
RUST_BACKTRACE=full cargo +nightly fuzz run --sanitizer=none <target> <target>/<crash-file>
```

---
*Auto-created by fuzzing workflow with Claude analysis*
```

## Key Features

### Smart Duplicate Detection

Claude compares crashes based on:
- **Crash location**: Same file and function (line numbers may differ)
- **Error pattern**: Same error type after normalizing values
  - Example: "index 5 out of bounds" = "index 12 out of bounds" (SAME)
  - Example: "len is 100" = "len is 5" (SAME)
- **Source code context**: Reads actual code to verify same root cause

### Occurrence Tracking

For duplicate crashes, Claude maintains a single tracking comment:

```markdown
<!-- occurrences: 15 -->
**Crash seen 15 time(s)**

Latest occurrence:
- Crash file: crash-abc123
- Artifact: [link]
- Branch: develop
- Commit: abc123
```

This keeps issues clean and shows crash frequency without spam.

### Conservative Approach

Claude is programmed to:
- Prefer creating new issues when unsure (avoid false positives)
- Focus on ROOT CAUSE rather than specific values in error messages
- Read source code to understand crashes deeply
- Only mark as duplicate with high confidence

## Claude's Capabilities

Claude has access to:

- ✅ Full repository source code (via `Read` tool)
- ✅ Fuzzer logs and crash output
- ✅ GitHub API (`gh` CLI for issues)
- ✅ Source code analysis at crash locations
- ✅ Cargo fuzz for crash reproduction (optional)

Claude uses:
- **Model**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Cost**: ~$0.03-0.05 per crash analysis
- **Max turns**: 25 (for complex analysis)

## Workflow Structure

```
┌─────────────────────────────────────────┐
│ io_fuzz / ops_fuzz                      │
│ - Run fuzzing target                    │
│ - Check for crashes                     │
│ - Archive artifacts + logs              │
│ - Output: crashes_found, first_crash   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ report-io-fuzz-failures /               │
│ report-ops-fuzz-failures                │
│ - Download fuzzer logs                  │
│ - Run Claude with analysis prompt       │
│ - Claude creates/updates issues         │
└─────────────────────────────────────────┘
```

## Configuration

### Required Secrets

- `CLAUDE_CODE_OAUTH_TOKEN` - OAuth token for Claude Code bot
- `R2_FUZZ_ACCESS_KEY_ID` - For fuzzer corpus storage
- `R2_FUZZ_SECRET_ACCESS_KEY` - For fuzzer corpus storage

### Required Labels

- `bug` - For bug reports
- `fuzzer` - For fuzzer-generated issues

### Workflow Permissions

**Fuzzing jobs (io_fuzz, ops_fuzz):**
- No special permissions needed (artifacts uploaded automatically)

**Reporting jobs (report-*-fuzz-failures):**
- `issues: write` - Create and update issues
- `contents: read` - Read repository code
- `id-token: write` - OIDC token for authentication
- `pull-requests: read` - Read PR context if needed

## Monitoring

### View Fuzzing Runs

All fuzzing runs are available at:
```
https://github.com/spiraldb/vortex/actions/workflows/fuzz.yml
```

### View Created Issues

All fuzzer-detected issues:
```
https://github.com/spiraldb/vortex/issues?q=is%3Aissue+label%3Afuzzer
```

### Artifacts

Each crash produces these artifacts (retained for 30 days):

1. **`<target>-fuzzing-crash-artifacts`** - All crash files found
2. **`<target>-fuzzing-logs`** - Complete fuzzer output with stack traces

## Customization

### Adding a New Fuzzing Target

1. Add a new job in `fuzz.yml` following the existing pattern (copy `io_fuzz`)
2. Add a corresponding `report-<target>-fuzz-failures` job
3. Update the target name in the Claude prompt

### Customizing Claude's Analysis

Edit the `prompt` in the `report-*-fuzz-failures` jobs:

```yaml
prompt: |
  # Fuzzer Crash Analysis and Reporting

  [Your custom instructions here]
```

### Changing the Model

Edit the `claude_args` section:

```yaml
claude_args: |
  --model claude-opus-4-20250514  # For more complex analysis
  --max-turns 25
  --allowedTools "..."
```

**Model Options:**
- `claude-sonnet-4-5-20250929` - Default, balanced cost/performance
- `claude-opus-4-20250514` - More capable, higher cost

### Adjusting Duplicate Detection Sensitivity

Modify the duplicate detection instructions in the prompt to make Claude more or less conservative about marking crashes as duplicates.

## Troubleshooting

### Claude Creates Too Many Duplicate Issues

Make Claude more aggressive about duplicate detection by:
- Emphasizing root cause comparison in the prompt
- Reducing the conservatism instruction
- Adding examples of what constitutes a duplicate

### Claude Misses Duplicates

Make Claude more thorough by:
- Increasing `--max-turns` to allow more analysis time
- Adding specific duplicate detection patterns for common errors
- Enhancing the error pattern normalization instructions

### Issues Not Created

Check:
1. `CLAUDE_CODE_OAUTH_TOKEN` is configured
2. Repository has `bug` and `fuzzer` labels
3. Reporting job has `issues: write` permission
4. Check workflow logs for errors

### Cannot Download Artifacts

Check:
1. Artifacts haven't expired (30-day retention)
2. Direct artifact URL from the issue is still valid
3. You have repository access

## Examples

### Example 1: New Crash Detected

1. Fuzzer detects crash in `file_io` target
2. `io_fuzz` job archives crash files and logs
3. `report-io-fuzz-failures` job triggered
4. Claude analyzes fuzzer log, identifies panic in `vortex_io::read_header`
5. Claude searches existing issues, finds no match
6. Claude creates new issue with detailed analysis
7. Issue includes stack trace, root cause, reproduction steps

### Example 2: Duplicate Crash

1. Fuzzer detects crash (same as issue #123)
2. Claude analyzes, recognizes same crash location and error pattern
3. Claude finds existing issue #123
4. Claude updates tracking comment: "Crash seen 5 time(s)"
5. No new issue created, keeping issue list clean

### Example 3: Similar but Different

1. Fuzzer detects crash in same function as issue #456
2. Claude analyzes, sees same location but different error pattern
3. Claude determines it's SIMILAR (medium confidence)
4. Claude adds comment to issue #456 explaining the similarity
5. Human reviews and decides if it's truly the same or needs new issue

## Best Practices

1. **Review Claude's Classifications** - Especially "similar" cases
2. **Close True Duplicates** - If Claude missed one, close and reference the original
3. **Add Labels** - Tag issues with severity (`P0`, `P1`, etc.)
4. **Track Frequency** - High occurrence counts indicate priority bugs
5. **Minimize Test Cases** - Use `cargo fuzz tmin` to create minimal reproducers
6. **Update Corpus** - Add interesting crashes to corpus after fixing

## Limitations

Claude may have difficulty with:

- ❌ Extremely complex crashes spanning multiple files
- ❌ Race conditions and timing-dependent bugs
- ❌ Crashes in dependencies (external crates)
- ❌ Crashes requiring deep domain knowledge

In these cases:
- Claude will still create an issue with available information
- Human analysis will be needed for root cause
- Claude's initial analysis can still help narrow the scope

## Cost and Performance

- **Analysis Time**: 1-3 minutes per crash
- **Cost**: ~$0.03-0.05 per crash (using Sonnet 4.5)
- **Accuracy**: High for duplicate detection (based on source code analysis)
- **False Positives**: Low (conservative by default)

## Future Enhancements

Potential improvements:

- [ ] Automatic crash minimization before reporting
- [ ] Severity classification (security vs stability)
- [ ] Automatic PR creation for simple fixes
- [ ] Integration with coverage reports
- [ ] Historical crash trend analysis
- [ ] Cross-target duplicate detection
- [ ] Automatic corpus optimization

## Support

For issues with the fuzzing infrastructure:
- Create an issue with the `fuzzing` label
- Tag relevant maintainers

For issues with Claude Code Action:
- See [Claude Code Action docs](https://github.com/anthropics/claude-code-action)
- Report at https://github.com/anthropics/claude-code-action/issues
