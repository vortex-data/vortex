# Automated Fuzzer Crash Analysis with Claude Code

This directory contains workflows for automated fuzzer crash detection and analysis using the Claude Code bot.

## Overview

The fuzzing infrastructure consists of two main workflows:

1. **`fuzz.yml`** - Runs fuzzing targets on a schedule and detects crashes
2. **`fuzzer-claude-analysis.yml`** - Automatically analyzes crashes using Claude Code

## How It Works

### 1. Crash Detection (fuzz.yml)

The fuzzing workflow runs every 4 hours and:

- Runs fuzzing targets (`file_io`, `array_ops`) for 2 hours each
- Detects crashes and saves crash artifacts
- Automatically creates a GitHub issue with:
  - Crash details and backtrace
  - Links to download crash artifacts
  - Reproduction instructions
  - Automatically mentions `@claude` to trigger analysis

**Fuzzing Targets:**
- `file_io` - Tests file I/O operations
- `array_ops` - Tests array operations

### 2. Automated Analysis (fuzzer-claude-analysis.yml)

When a fuzzer crash issue is created, the Claude Code bot automatically:

1. **Downloads** crash artifacts from the workflow run
2. **Reproduces** the crash locally with full backtrace
3. **Analyzes** the crash to identify the root cause
4. **Assesses** severity (security issue, panic, or logic error)
5. **Attempts to fix** the issue if possible
6. **Creates regression tests** to prevent recurrence
7. **Posts findings** as a comment on the issue

### 3. Triggering the Analysis

The Claude analysis workflow is triggered when:

- An issue is created with the `fuzzing` label, OR
- An issue title contains "Fuzzing Crash Report"

The fuzzing workflow automatically creates issues that meet these criteria.

## Manual Triggering

You can also manually trigger Claude analysis on any fuzzer-related issue by:

1. Adding the `fuzzing` label to the issue
2. Mentioning `@claude` in a comment with specific instructions

Example:
```
@claude - Please analyze this fuzzer crash and create a fix if possible.
```

## Issue Format

Fuzzer crash issues are automatically created with:

```markdown
## Fuzzing Crash Report

The `<target>` fuzzing target detected a <crash_type> during a scheduled fuzzing run.

### Summary
- **Crash Type**: crash/panic/timeout
- **Target**: `file_io` or `array_ops`
- **Crash File**: `crash-<hash>`
- **Workflow Run**: [link]
- **Timestamp**: [UTC timestamp]

### Crash Artifacts
[Download links to artifacts]

### Reproduction Steps
[Commands to reproduce locally]

### Investigation Checklist
[Checklist of investigation steps]

### Automated Analysis
@claude - Please analyze this fuzzer crash:
1. Download and reproduce the crash
2. Identify the root cause
3. Assess severity
4. Create a fix if possible
5. Verify the fix

### Environment
[Environment details]
```

## Claude's Analysis Process

When triggered, Claude will:

1. **Setup Environment**
   - Checkout repository
   - Install Rust nightly toolchain
   - Install cargo-fuzz

2. **Reproduce Crash**
   - Download crash artifacts from the workflow run
   - Run the fuzzer with the crash file
   - Capture full backtrace with `RUST_BACKTRACE=full`

3. **Analyze Root Cause**
   - Examine backtrace and crash output
   - Identify the code path that triggered the crash
   - Determine the underlying issue

4. **Severity Assessment**
   - **Security**: Memory safety issues, overflows, out-of-bounds access
   - **Panic**: Unwraps, assertions, explicit panics
   - **Logic Error**: Incorrect behavior without panic

5. **Create Fix** (if possible)
   - Modify the relevant source files
   - Add input validation or error handling
   - Follow project code style guidelines

6. **Write Regression Tests**
   - Create test cases that capture the failure
   - Use the minimized crash file as test input
   - Ensure tests fail before fix and pass after

7. **Verify Fix**
   - Run fuzzer again with the crash file
   - Run full test suite
   - Check with clippy for any new warnings

8. **Post Results**
   - Comment on the issue with findings
   - If a fix was created, push a branch and reference the issue
   - Provide next steps for manual review

## What Claude Can Do

Claude has access to:

- ✅ All project source code
- ✅ Full cargo toolchain (build, test, clippy, fmt, fuzz)
- ✅ Crash artifacts and backtrace
- ✅ Git operations (create branches, commit changes)
- ✅ File editing capabilities

Claude will:

- ✅ Follow project code style (see `CLAUDE.md`)
- ✅ Run `cargo clippy` and `cargo fmt` before finishing
- ✅ Create meaningful regression tests
- ✅ Provide detailed explanations of the root cause

## Limitations

Claude may not be able to fix:

- ❌ Issues requiring extensive architectural changes
- ❌ Issues in dependencies or external crates
- ❌ Complex concurrency bugs
- ❌ Issues requiring domain-specific knowledge

In these cases, Claude will still provide:
- Root cause analysis
- Severity assessment
- Suggestions for how to approach the fix

## Workflow Permissions

The workflows require these GitHub permissions:

**fuzz.yml:**
- `contents: read` - Read repository code
- `issues: write` - Create crash report issues

**fuzzer-claude-analysis.yml:**
- `contents: write` - Create branches with fixes
- `pull-requests: write` - Create PRs (if needed)
- `issues: write` - Comment on issues
- `actions: read` - Download artifacts from workflow runs

## Configuration

### Required Secrets

- `CLAUDE_CODE_OAUTH_TOKEN` - OAuth token for Claude Code bot
- `R2_FUZZ_ACCESS_KEY_ID` - For fuzzer corpus storage (existing)
- `R2_FUZZ_SECRET_ACCESS_KEY` - For fuzzer corpus storage (existing)

### Modifying the Workflows

**To add a new fuzzing target:**

1. Add a new job to `fuzz.yml` following the existing pattern
2. Update `.github/scripts/create-fuzzer-issue.sh` if needed
3. The Claude analysis workflow will automatically handle it

**To customize Claude's behavior:**

Edit the `custom_instructions` in `fuzzer-claude-analysis.yml`:

```yaml
custom_instructions: |
  You are analyzing a fuzzer crash. Focus on:
  1. <your custom instruction>
  2. <another instruction>
```

**To change the model:**

The workflow uses Claude Opus 4 for complex crash analysis. To use a different model:

```yaml
model: "claude-sonnet-4-20250514"  # Faster, lower cost
# or
model: "claude-opus-4-20250514"    # Most capable, higher cost
```

## Monitoring

### View Fuzzing Runs

All fuzzing runs are available at:
```
https://github.com/spiraldb/vortex/actions/workflows/fuzz.yml
```

### View Claude Analysis Runs

All Claude analysis runs are available at:
```
https://github.com/spiraldb/vortex/actions/workflows/fuzzer-claude-analysis.yml
```

### Artifacts

Each crash produces these artifacts:

1. **From fuzzing workflow:**
   - `<target>-fuzzing-crash-artifacts` - All crash files
   - `<target>-fuzzing-logs` - Fuzzer output logs

2. **From Claude analysis:**
   - All analysis results are posted directly as comments on the issue
   - The crash output and backtrace are included in the issue comments
   - Full workflow logs available in the GitHub Actions run

## Troubleshooting

### Issue Not Created After Crash

Check:
1. Fuzzing workflow has `issues: write` permission
2. GitHub CLI (`gh`) is available in the runner
3. Check workflow logs for error messages

### Claude Not Analyzing Issue

Check:
1. Issue has `fuzzing` label OR title contains "Fuzzing Crash Report"
2. `CLAUDE_CODE_OAUTH_TOKEN` secret is configured
3. Claude workflow has required permissions

### Cannot Download Artifacts

Check:
1. Workflow run ID in the issue is correct
2. Artifacts haven't expired (GitHub retains for 90 days)
3. Claude workflow has `actions: read` permission

### Crash Cannot Be Reproduced

This can happen if:
- The crash was environment-specific (timing, resources)
- The crash has already been fixed in main branch
- Artifacts are corrupted or incomplete

Claude will note this in the analysis comment.

## Examples

### Successful Analysis

When Claude successfully analyzes and fixes a crash:

1. Issue is created automatically by fuzzing workflow
2. Claude workflow triggers within minutes
3. Initial comment is posted with:
   - ✅ Crash reproduction confirmation
   - 📋 Full backtrace and crash output
   - 🔧 Reproduction command
4. Claude analyzes and posts follow-up comments with:
   - ✅ Root cause identified
   - ✅ Severity assessment
   - ✅ Fix created in branch `fuzzer-fix-<target>-<issue-number>`
   - ✅ Regression tests added
   - ✅ Verification results
5. A human reviews the fix and creates a PR

### Partial Analysis

When Claude can analyze but not fix:

1. Issue is created
2. Initial comment shows crash reproduction
3. Claude posts follow-up comments with:
   - ✅ Root cause identified
   - ✅ Severity assessment
   - ❌ Fix not created (explanation provided)
   - 📝 Suggestions for fixing
4. A human uses Claude's analysis to create a fix

All findings are posted directly to the issue for easy review and discussion.

## Best Practices

1. **Review all Claude fixes** - Even successful fixes should be manually reviewed
2. **Check regression tests** - Ensure tests actually capture the failure
3. **Verify across platforms** - Test fixes on different architectures if relevant
4. **Update fuzzer corpus** - Add minimized crash files to corpus if valuable
5. **Close duplicates** - Check if similar crashes have been reported

## Future Enhancements

Potential improvements:

- [ ] Automatic PR creation by Claude (currently creates branch only)
- [ ] Crash deduplication before issue creation
- [ ] Integration with coverage reports
- [ ] Automatic corpus minimization
- [ ] Historical crash trend analysis
- [ ] Severity-based issue prioritization

## Support

For issues with the fuzzing infrastructure:
- Create an issue with the `fuzzing` label
- Tag relevant maintainers

For issues with Claude Code:
- See [Claude Code documentation](https://github.com/anthropics/claude-code-action)
- Report issues at https://github.com/anthropics/claude-code/issues
