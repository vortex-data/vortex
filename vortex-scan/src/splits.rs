// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use crate::selection::Selection;

/// The way in which we compute splits for a file.
pub(super) enum Splits {
    /// Natural splits computed by the layout reader (e.g., computing splits across different-sized
    /// column chunks).
    Natural(BTreeSet<u64>),

    /// Exact split ranges. This is an optimization for when we know the exact rows we need to get
    /// from a file (which is common if we just want to call `take` with a few sparse indices).
    Ranges(Vec<Range<u64>>),
}

/// Attempts to compute split ranges from the given selection.
///
/// TODO more docs.
pub(super) fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };

    // TODO(connor): We can be smarter here, as the row range is more restrictive than the
    // selection.
    if row_range.is_some() {
        return None;
    }

    // We want to create ranges that will represent splits that cover our indices.
    // We want to make sure that we do not create too many splits. We also want to make sure our
    // splits do not cover too much as they would overlap column chunk boundaries.

    None
}
