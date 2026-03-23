// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use humansize::DECIMAL;
use humansize::format_size;

use crate::DynArray;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds `nbytes=X (Y%)` to the header line.
pub struct NbytesExtractor;

impl TreeExtractor for NbytesExtractor {
    fn header_annotations(&self, array: &dyn DynArray, ctx: &TreeContext) -> Vec<String> {
        let nbytes = array.nbytes();
        let total_size = ctx.parent_total_size().unwrap_or(nbytes);
        let percent = if total_size == 0 {
            0.0
        } else {
            100_f64 * nbytes as f64 / total_size as f64
        };
        vec![format!(
            "nbytes={} ({:.2}%)",
            format_size(nbytes, DECIMAL),
            percent
        )]
    }
}
