// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expected row counts per origin, captured by `sqlstorm-select` (a later task).
//! `None` until that task populates them.

use super::SqlstormOrigin;

pub fn expected_row_counts(origin: SqlstormOrigin) -> Option<Vec<usize>> {
    let _ = origin;
    None
}
