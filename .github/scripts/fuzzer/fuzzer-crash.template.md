## Fuzzing Crash Report

### Crash Summary

| Field | Value |
|-------|-------|
| **Target** | `{{FUZZ_TARGET}}` |
| **Crash Type** | `{{CRASH_TYPE}}` |
| **Crash File** | `{{CRASH_FILE}}` |
| **Seed Hash** | `{{SEED_HASH}}` |
| **Stack Hash** | `{{STACK_HASH}}` |
| **Message Hash** | `{{MESSAGE_HASH}}` |

### Location

**Panic Location**: `{{PANIC_LOCATION}}`

**Error Variant**: `{{ERROR_VARIANT}}`

### Error Message

```
{{PANIC_MESSAGE}}
```

### Stack Trace

```
{{STACK_TRACE}}
```

### Claude Analysis

{{CLAUDE_ANALYSIS}}

<details>
<summary>Debug Output</summary>

```
{{DEBUG_OUTPUT}}
```

</details>

### Build Info

| Field | Value |
|-------|-------|
| **Branch** | {{BRANCH}} |
| **Commit** | {{COMMIT}} |
| **Artifact URL** | {{ARTIFACT_URL}} |

### Reproduction

1. Download the crash artifact from the link above
2. Extract and run:

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

3. For full backtrace:

```bash
RUST_BACKTRACE=full cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

---
*Auto-created by fuzzing workflow*
