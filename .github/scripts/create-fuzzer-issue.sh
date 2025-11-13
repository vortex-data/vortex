#!/bin/bash

set -euo pipefail

# Script to create a GitHub issue for a fuzzer crash
# Usage: ./create-fuzzer-issue.sh <target> <crash_type> <artifact_name>

TARGET="${1:-unknown}"
CRASH_TYPE="${2:-crash}"
ARTIFACT_NAME="${3:-fuzzing-crash-artifacts}"
WORKFLOW_RUN="${GITHUB_RUN_ID:-unknown}"
CRASH_COUNT="${4:-1}"

# Get the first crash file name if available
CRASH_FILE=$(find fuzz/artifacts/$TARGET -name "crash-*" -type f | head -1 | xargs basename || echo "crash-unknown")

# Get current timestamp
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Get current branch and commit
BRANCH="${GITHUB_REF_NAME:-unknown}"
COMMIT="${GITHUB_SHA:-unknown}"

# Create issue title
ISSUE_TITLE="[Fuzzer] ${CRASH_TYPE} in ${TARGET} target"

# Create issue body
ISSUE_BODY=$(cat <<EOF
## Fuzzing Crash Report

The \`${TARGET}\` fuzzing target detected a ${CRASH_TYPE} during a scheduled fuzzing run.

### Summary

- **Crash Type**: ${CRASH_TYPE}
- **Target**: \`${TARGET}\`
- **Crash File**: \`${CRASH_FILE}\`
- **Total Crashes Found**: ${CRASH_COUNT}
- **Workflow Run**: ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${WORKFLOW_RUN}
- **Timestamp**: ${TIMESTAMP}
- **Branch**: ${BRANCH}
- **Commit**: ${COMMIT}

### Crash Artifacts

Download crash artifacts from the workflow run:
**${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${WORKFLOW_RUN}**

Artifacts available:
- \`${ARTIFACT_NAME}\` - All crash files found (includes ${CRASH_COUNT} crashes)
- \`${TARGET}-fuzzing-logs\` - Complete fuzzer output with stack traces

### Reproduction Steps

1. Download the \`${ARTIFACT_NAME}\` from the workflow run above
2. Extract the crash file to your local \`fuzz/artifacts/${TARGET}/\` directory
3. Reproduce the crash locally:

\`\`\`bash
cargo +nightly fuzz run ${TARGET} fuzz/artifacts/${TARGET}/${CRASH_FILE}
\`\`\`

4. Get full backtrace:

\`\`\`bash
RUST_BACKTRACE=full cargo +nightly fuzz run ${TARGET} fuzz/artifacts/${TARGET}/${CRASH_FILE}
\`\`\`

5. Minimize the test case (optional):

\`\`\`bash
cargo +nightly fuzz tmin ${TARGET} fuzz/artifacts/${TARGET}/${CRASH_FILE}
\`\`\`

### Investigation Checklist

- [ ] Download crash artifacts from workflow run
- [ ] Reproduce crash locally with full backtrace
- [ ] Analyze stack trace and identify root cause
- [ ] Determine severity (security vs stability)
- [ ] Check if this is a duplicate of an existing issue
- [ ] Minimize test case if needed
- [ ] Create fix PR with reference to this issue
- [ ] Add regression test
- [ ] Verify fix with: \`cargo +nightly fuzz run ${TARGET} <crash-file>\`

### Automated Analysis

@claude - Please analyze this fuzzer crash:
1. Download and reproduce the crash
2. Identify the root cause from the backtrace
3. Assess the severity (security issue, panic, or logic error)
4. If possible, create a fix and regression tests
5. Verify the fix resolves the crash

### Environment

- **Runner**: ${RUNNER_OS:-ubuntu}
- **Rust Toolchain**: nightly
- **Fuzz Duration**: 7200 seconds (2 hours)
- **Fuzzer**: cargo-fuzz (libFuzzer)

### Additional Context

This issue was automatically created by the fuzzing workflow. If multiple crashes were found, this issue represents the first crash detected. All crash artifacts are available for download.

**Note**: If this issue is a duplicate of an existing bug, please close it and reference the original issue.

---

*Automatically created by fuzzing workflow*
*Workflow file: [fuzz.yml](${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/blob/${BRANCH}/.github/workflows/fuzz.yml)*
*Run: ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${WORKFLOW_RUN}*
EOF
)

# Create the issue using GitHub CLI
if command -v gh &> /dev/null; then
    echo "Creating GitHub issue..."
    gh issue create \
        --title "$ISSUE_TITLE" \
        --body "$ISSUE_BODY" \
        --label "bug,fuzzing" \
        --assignee "@me"

    echo "Issue created successfully!"
else
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Issue body would have been:"
    echo "$ISSUE_BODY"
    exit 1
fi
