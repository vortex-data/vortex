// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::dtype::DType;

/// Extractor that adds the encoding summary (e.g. `vortex.primitive(i16, len=5)`) to the header.
pub struct EncodingSummaryExtractor;

impl EncodingSummaryExtractor {
    /// Write the encoding summary for an array directly to a formatter.
    pub fn write(array: &ArrayRef, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::write_parts(array.encoding_id(), array.dtype(), array.len(), f)
    }

    /// Write the encoding summary from its constituent parts.
    ///
    /// Callers that hold a parent's metadata without an [`ArrayRef`] — a
    /// [`ParentRef`](crate::array::ParentRef) borrowing construction parts, or a captured
    /// summary — use this to render the same `vortex.primitive(i16, len=5)` form as every
    /// other array print.
    pub fn write_parts(
        encoding_id: ArrayId,
        dtype: &DType,
        len: usize,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{encoding_id}({dtype}, len={len})")
    }
}

impl TreeExtractor<ArrayRef, TreeContext> for EncodingSummaryExtractor {
    fn write_header(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, " ")?;
        Self::write(array, f)
    }
}
