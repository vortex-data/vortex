// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Global tracing and instrumentation

use parking_lot::RwLock;

static LABELS: RwLock<Vec<(&str, String)>> = RwLock::new(Vec::new());

/// Set global labels used for external tracing.
pub fn set_global_labels<I>(i: I)
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    let new = Vec::from_iter(i);
    *LABELS.write() = new;
}

/// Get the globally set labels for external tracing.
pub fn get_global_labels() -> Vec<(&'static str, String)> {
    LABELS.read().clone()
}
