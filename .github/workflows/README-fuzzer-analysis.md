# Automated Fuzzer Crash Analysis with Claude Code

This directory contains workflows for automated fuzzer crash detection, analysis, issue creation, and fix automation using the Claude Code bot.

## Overview

The fuzzing infrastructure has **two stages**:

1. **Stage 1 (fuzz.yml)**: Detects crashes and uses Claude to analyze and create/update GitHub issues with smart duplicate detection
2. **Stage 2 (fuzzer-fix-automation.yml)**: When a fuzzer issue is created, automatically attempts to fix it, create regression tests, and post findings

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

**Stage 1 (Issue Creation)** uses:
- **Model**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Cost**: ~$0.03-0.05 per crash analysis
- **Max turns**: 25 (for complex analysis)

**Stage 2 (Fix Automation)** uses:
- **Model**: Claude Opus 4 (`claude-opus-4-20250514`) - more capable for code generation
- **Cost**: ~$0.15-0.25 per fix attempt
- **Max turns**: 40 (allows for iterative fixing and testing)

### 3. Automated Fix Attempt (fuzzer-fix-automation.yml)

When a fuzzer issue is created (labeled with `fuzzer`), Claude automatically:

1. **Extracts Crash Details** - Parses the issue body for:
   - Target name
   - Crash file name
   - Artifact download URL
   - Stack trace and error message

2. **Downloads and Reproduces** - Attempts to:
   - Download the crash artifact
   - Reproduce the crash locally with the fuzzer
   - Verify the panic/error occurs

3. **Analyzes Root Cause** - Deep analysis of:
   - Source code at crash location
   - Stack trace to understand call path
   - Debug output to see problematic input
   - Determines the underlying bug

4. **Assesses Fixability** - Decides if this is fixable automatically:
   - **CAN FIX**: Missing bounds check, validation, edge case handling, simple panics
   - **CANNOT FIX**: Architectural issues, complex logic, requires domain knowledge

5. **Creates Fix (if straightforward)**:
   - Modifies source code with minimal changes
   - Adds validation or bounds checks
   - Handles the edge case properly
   - Follows project code style guidelines

6. **Writes Regression Tests**:
   - Creates test using the actual fuzzer input that triggered the crash
   - Test fails before the fix, passes after
   - Placed in appropriate test module
   - Named clearly (e.g., `test_fuzzer_crash_issue_123`)

7. **Verifies the Fix**:
   - Runs regression test
   - Runs fuzzer with crash file (should not panic)
   - Runs related tests
   - Checks with clippy
   - Formats code

8. **Posts Findings** - Comments on the issue with:
   - Root cause analysis
   - Fix description (if created)
   - Regression test details
   - Verification results
   - OR explanation of why it can't be fixed automatically

## Workflow Structure

```
┌─────────────────────────────────────────┐
│ STAGE 1: Detection & Issue Creation    │
│ (fuzz.yml)                              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ io_fuzz / ops_fuzz                      │
│ - Run fuzzing target (2 hours)          │
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
│   • Smart duplicate detection           │
│   • Occurrence tracking                 │
│   • Detailed crash analysis             │
└──────────────┬──────────────────────────┘
               │
               │ Issue created with 'fuzzer' label
               │
               ▼
┌─────────────────────────────────────────┐
│ STAGE 2: Automated Fix Attempt         │
│ (fuzzer-fix-automation.yml)             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ attempt-fix                             │
│ - Triggered by issue with 'fuzzer' label│
│ - Download crash artifact               │
│ - Reproduce the crash                   │
│ - Analyze root cause                    │
│ - Create fix if straightforward         │
│ - Write regression tests                │
│ - Verify fix works                      │
│ - Post analysis comment                 │
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

**Fix automation job (fuzzer-fix-automation.yml):**
- `contents: write` - Modify source files to create fixes
- `pull-requests: write` - Create PRs if requested
- `issues: write` - Comment on issues with findings
- `id-token: write` - OIDC token for authentication

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

### Example 1: New Crash Detected and Auto-Fixed

**Stage 1 - Issue Creation:**
1. Fuzzer detects crash in `file_io` target: "index out of bounds"
2. `io_fuzz` job archives crash files and logs
3. `report-io-fuzz-failures` job triggered
4. Claude analyzes fuzzer log, identifies panic in `vortex_io::read_header` at line 45
5. Claude searches existing issues, finds no match
6. Claude creates issue #789 with detailed analysis, labels it `bug,fuzzer`

**Stage 2 - Fix Automation:**
7. `fuzzer-fix-automation` workflow triggers on issue #789
8. Claude extracts crash details from issue body
9. Claude downloads crash artifact
10. Claude reproduces the crash locally
11. Claude analyzes source code at `vortex_io::read_header:45`
12. Root cause: Missing bounds check before indexing into buffer
13. Claude creates fix: Adds validation `if index >= buffer.len() { return Err(...) }`
14. Claude writes regression test: `test_fuzzer_crash_issue_789()`
15. Claude verifies: test passes, fuzzer doesn't crash, clippy passes
16. Claude comments on issue #789 with full analysis and fix details
17. Human reviews and merges the fix

### Example 2: Duplicate Crash (No Fix Attempted)

**Stage 1 - Duplicate Detection:**
1. Fuzzer detects crash (same as issue #123)
2. Claude analyzes, recognizes same crash location and error pattern
3. Claude finds existing issue #123
4. Claude updates tracking comment: "Crash seen 5 time(s)"
5. No new issue created, keeping issue list clean

**Stage 2 - No trigger:**
6. Fix automation doesn't trigger (no new issue created)
7. Human can manually trigger on issue #123 if desired

### Example 3: Complex Crash (Analysis Only)

**Stage 1 - Issue Creation:**
1. Fuzzer detects crash in `array_ops` target
2. Claude creates issue #790 with analysis

**Stage 2 - Cannot Auto-Fix:**
3. `fuzzer-fix-automation` triggers on issue #790
4. Claude analyzes the crash
5. Determines it's an architectural issue requiring refactoring
6. Claude comments: "This requires human intervention" with detailed analysis
7. Provides suggestions for how to approach the fix
8. Human developer takes over from Claude's analysis

## Best Practices

### For Stage 1 (Issue Creation)

1. **Review Claude's Classifications** - Especially "similar" cases
2. **Close True Duplicates** - If Claude missed one, close and reference the original
3. **Add Labels** - Tag issues with severity (`P0`, `P1`, etc.)
4. **Track Frequency** - High occurrence counts indicate priority bugs

### For Stage 2 (Fix Automation)

1. **Review All Fixes** - Claude's fixes are suggestions, always review before merging
2. **Test Thoroughly** - Run the regression test and broader test suite
3. **Check Edge Cases** - Verify Claude considered all edge cases, not just the crash
4. **Assess Test Quality** - Ensure regression tests actually catch the bug
5. **Consider Broader Impact** - Check if the same issue exists elsewhere in the codebase
6. **Minimize Test Cases** - Use `cargo fuzz tmin` to create minimal reproducers
7. **Update Corpus** - Add interesting crashes to corpus after fixing
8. **Close Issues** - Once merged, close the issue and reference the PR

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

### Stage 1 (Issue Creation)
- **Analysis Time**: 1-3 minutes per crash
- **Cost**: ~$0.03-0.05 per crash (using Sonnet 4.5)
- **Accuracy**: High for duplicate detection (based on source code analysis)
- **False Positives**: Low (conservative by default)

### Stage 2 (Fix Automation)
- **Analysis Time**: 5-15 minutes per crash (includes reproduction, analysis, fixing, testing)
- **Cost**: ~$0.15-0.25 per fix attempt (using Opus 4)
- **Success Rate**: Depends on crash complexity
  - Simple bugs (bounds checks, validation): ~70-80% fix rate
  - Medium complexity: ~30-50% fix rate
  - Complex bugs: Analysis only, human intervention needed
- **False Fixes**: Very low (Claude is conservative about committing changes)

## Future Enhancements

### Stage 1 Enhancements
- [ ] Automatic crash minimization before reporting
- [ ] Integration with coverage reports
- [ ] Historical crash trend analysis
- [ ] Cross-target duplicate detection
- [ ] Automatic corpus optimization

### Stage 2 Enhancements
- [ ] Automatic PR creation (currently just posts fix)
- [ ] Severity classification (security vs stability)
- [ ] Suggest fixes to similar code patterns across codebase
- [ ] Batch fix multiple similar crashes
- [ ] Learn from accepted/rejected fixes to improve future attempts

## Support

For issues with the fuzzing infrastructure:
- Create an issue with the `fuzzing` label
- Tag relevant maintainers

For issues with Claude Code Action:
- See [Claude Code Action docs](https://github.com/anthropics/claude-code-action)
- Report at https://github.com/anthropics/claude-code-action/issues
