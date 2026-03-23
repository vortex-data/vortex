// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::DynArray;
use crate::display::DisplayOptions;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds the encoding summary (e.g. `vortex.primitive(i16, len=5)`) to the header.
pub struct EncodingSummaryExtractor;

impl TreeExtractor for EncodingSummaryExtractor {
    fn header_annotations(&self, array: &dyn DynArray, _ctx: &TreeContext) -> Vec<String> {
        vec![format!(
            "{}",
            array.display_as(DisplayOptions::MetadataOnly)
        )]
    }
}
