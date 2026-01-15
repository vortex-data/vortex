## Related Crash Detected

A similar crash was detected matching this issue.

### Match Details

| Field | Value |
|-------|-------|
| **Match Type** | {{MATCH_CHECK}} |
| **Confidence** | {{MATCH_CONFIDENCE}} |
| **Reason** | {{MATCH_REASON}} |

### New Crash Summary

| Field | Value |
|-------|-------|
| **Target** | `{{FUZZ_TARGET}}` |
| **Crash Type** | `{{CRASH_TYPE}}` |
| **Crash File** | `{{CRASH_FILE}}` |

<details>
<summary>Deduplication Hashes</summary>

| Hash Type | Value |
|-----------|-------|
| **Seed Hash** | `{{SEED_HASH}}` |
| **Stack Hash** | `{{STACK_HASH}}` |

</details>

### Location

**Panic Location**: `{{PANIC_LOCATION}}`

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

```bash
cargo +nightly fuzz run -D --sanitizer=none {{FUZZ_TARGET}} {{FUZZ_TARGET}}/{{CRASH_FILE}} -- -rss_limit_mb=0
```

---
*Auto-detected by fuzzing workflow*
