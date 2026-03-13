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
<summary>First-time setup: download and extract the crash artifact</summary>

1. Download the crash artifact:
   - **Direct download**: {{ARTIFACT_URL}}
   - Extract the zip file (`unzip`)
     - The path should look like `/path/to/{{FUZZ_TARGET}}/{{CRASH_FILE}}`
     - You can create a `./fuzz/artifacts` directory that will be git-ignored in the `vortex` repo
     - Full path would be `./fuzz/artifacts/{{FUZZ_TARGET}}/{{CRASH_FILE}}`

2. Assuming you download the zipfile to `~/Downloads`, and your working directory is the repository root:

```bash
mkdir -p ./fuzz/artifacts
mv ~/Downloads/{{FUZZ_TARGET}}-crash-artifacts.zip ./fuzz/artifacts/
unzip ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip -d ./fuzz/artifacts/
rm ./fuzz/artifacts/{{FUZZ_TARGET}}-crash-artifacts.zip
```

3. Get a backtrace:

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
