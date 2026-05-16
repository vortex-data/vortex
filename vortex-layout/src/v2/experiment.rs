// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporary runtime knobs for V2 performance experiments.
//!
//! These let benchmark runs isolate scheduler, row-demand, and
//! batching effects without recompiling between variants.

use std::env;
use std::sync::LazyLock;

pub(crate) fn bool_var(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(crate) fn usize_var(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub(crate) fn trace_flow() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| bool_var("VORTEX_V2_TRACE_FLOW"));
    *ENABLED
}
