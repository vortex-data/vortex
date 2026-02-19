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

1. Download the crash artifact:
   - **Direct download**: {{ARTIFACT_URL}}
   - Extract the zip file

2. Reproduce locally:
```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

3. Get full backtrace:
```bash
RUST_BACKTRACE=full cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

<!-- seed_hash:{{SEED_HASH}} stack_hash:{{STACK_TRACE_HASH}} message_hash:{{MESSAGE_HASH}} -->

---
*Auto-created by fuzzing workflow*
