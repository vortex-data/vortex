// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! V1/V2 scan toggle.
//!
//! While the LayoutPlan v2 path is being built out, the legacy
//! `LayoutReader`-driven scan remains the default. Set
//! `VORTEX_LAYOUT_PLAN_V2=1` to route compatible queries through the
//! new v2 path. Unsupported query shapes fall back to v1.
//!
//! This is a temporary mechanism. It will be removed once v2 reaches
//! parity and the default is flipped.

use std::env;

/// Environment variable that opts into the v2 scan path.
pub const ENV_VAR: &str = "VORTEX_LAYOUT_PLAN_V2";

/// True iff the v2 scan toggle is set via [`ENV_VAR`]. Recognised
/// truthy values: `1`, `true`, `yes`, `on` (case-insensitive).
pub fn use_v2_scan() -> bool {
    match env::var(ENV_VAR) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We deliberately don't test env-var behaviour here — env-var
    // access is a process-global side effect that's racy under
    // `cargo test`'s parallel execution. The helper is trivial
    // enough that a unit test would be all mock anyway.

    #[test]
    fn env_var_name_is_stable() {
        // Regression check: code elsewhere reads VORTEX_LAYOUT_PLAN_V2
        // directly (e.g., benchmark setup scripts).
        assert_eq!(ENV_VAR, "VORTEX_LAYOUT_PLAN_V2");
    }

    #[test]
    fn no_env_var_is_false() {
        // Save / restore is omitted on purpose; this test relies on
        // the env var being unset in CI. Tests for truthy values are
        // not included since toggling env vars across parallel tests
        // is unsafe in modern Rust.
        // SAFETY: tests run in the same process; setting/unsetting
        // env vars between tests is racy. We only assert behaviour
        // when the var is unset, which is the default state.
        if env::var(ENV_VAR).is_err() {
            assert!(!use_v2_scan());
        }
    }
}
