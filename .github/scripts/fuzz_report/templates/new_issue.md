## Fuzzing Crash Report

### Analysis

**Crash Location**: `{{CRASH_LOCATION}}`

**Error Message**:

```
{{PANIC_MESSAGE}}
```

<details>
<summary>Stack Trace</summary>

```
{{STACK_TRACE_RAW}}
```

</details>
{% if CLAUDE_ANALYSIS %}

### Root Cause Analysis

{{CLAUDE_ANALYSIS}}
{% endif %}

### Summary

- **Target**: `{{FUZZ_TARGET}}`
- **Crash File**: `{{CRASH_FILE}}`
- **Branch**: {{BRANCH}}
- **Commit**: {{COMMIT}}
- **Crash Artifact**: {{ARTIFACT_URL}}

### Reproduction

<details>

1. Download the crash artifact:
   - **Direct download**: {{ARTIFACT_URL}}
   - Extract the zip file (`unzip`)
     - The path should look like `/path/to/{{FUZZ_TARGET}}/{{CRASH_FILE}}`
     - You can create a `./fuzz/artifacts` directory that will be git-ignored in the `vortex` repo
     - Full path would be `./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}}`

2. Reproduce locally:

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} /path/to/crash_file -- -rss_limit_mb=0
```

3. Get a backtrace:

```bash
RUST_BACKTRACE=1 cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} /path/to/crash_file -- -rss_limit_mb=0
```

```bash
RUST_BACKTRACE=full cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} /path/to/crash_file -- -rss_limit_mb=0
```

</details>

### Workflow Example

Assuming you download the zipfile to `~/Downloads`, and your working directory is the repository
root, you can follow these steps:

<details>

```bash
# Create the artifacts directory if you haven't already.
mkdir -p ./fuzz/artifacts

# Move the zipfile.
mv ~/Downloads/{{FUZZ_TARGET}}-crash-artifacts.zip ./fuzz/artifacts/

# Unzip the zipfile.
unzip ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip -d ./fuzz/artifacts/

# You can remove the zipfile now if you want to.
rm ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip
```

You can now reproduce with:

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} ./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

If you want a backtrace:

```bash
RUST_BACKTRACE=1 cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} ./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

```bash
RUST_BACKTRACE=full cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} ./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

</details>

<!-- seed_hash:{{SEED_HASH}} stack_hash:{{STACK_TRACE_HASH}} message_hash:{{MESSAGE_HASH}} -->

---

_Auto-created by fuzzing workflow_
