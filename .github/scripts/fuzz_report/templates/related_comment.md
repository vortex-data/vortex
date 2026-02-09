## Related Crash Detected

A similar crash was detected in the `{{FUZZ_TARGET}}` target.

**Match**: {{DEDUP_REASON}} (confidence: {{DEDUP_CONFIDENCE}})

### Crash Details

**Crash Location**: `{{CRASH_LOCATION}}`

**Error Message**:
```
{{PANIC_MESSAGE}}
```

**Stack Trace**:
```
{{STACK_TRACE_RAW}}
```
{{#if DEBUG_OUTPUT}}

<details>
<summary>Debug Output</summary>

```
{{DEBUG_OUTPUT}}
```
</details>
{{/if}}

### Occurrence Details

- **Target**: `{{FUZZ_TARGET}}`
- **Crash File**: `{{CRASH_FILE}}`
- **Branch**: {{BRANCH}}
- **Commit**: {{COMMIT}}
- **Crash Artifact**: {{ARTIFACT_URL}}

### Reproduction

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

<!-- seed_hash:{{SEED_HASH}} stack_hash:{{STACK_TRACE_HASH}} message_hash:{{MESSAGE_HASH}} -->

---
*Auto-detected by fuzzing workflow*
