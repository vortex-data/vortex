// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::DynArray;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds the encoding summary (e.g. `vortex.primitive(i16, len=5)`) to the header.
pub struct EncodingSummaryExtractor;

impl EncodingSummaryExtractor {
    /// Write the encoding summary for an array directly to a formatter.
    pub fn write(array: &dyn DynArray, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}, len={})",
            array.encoding_id(),
            array.dtype(),
            array.len()
        )
    }
}

impl TreeExtractor for EncodingSummaryExtractor {
    fn write_header(
        &self,
        array: &dyn DynArray,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, " ")?;
        Self::write(array, f)
    }
}
