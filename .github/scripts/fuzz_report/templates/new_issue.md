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

### Reproduce

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} ./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

<details>
<summary>Reproduction Steps</summary>

1. Download the crash artifact: {{ARTIFACT_URL}}

2. Assuming you download the zipfile to `~/Downloads`, and your working directory is the repository root:

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

3. Reproduce the crash:

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

<details>
<summary>Single command to get a backtrace</summary>

```bash
mkdir -p ./fuzz/artifacts
mv ~/Downloads/{{FUZZ_TARGET}}-crash-artifacts.zip ./fuzz/artifacts/
unzip ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip -d ./fuzz/artifacts/
rm ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip
RUST_BACKTRACE=1 cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} ./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

</details>

<!-- seed_hash:{{SEED_HASH}} stack_hash:{{STACK_TRACE_HASH}} message_hash:{{MESSAGE_HASH}} -->

---

_Auto-created by fuzzing workflow_
