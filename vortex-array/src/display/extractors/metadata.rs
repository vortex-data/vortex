// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;
use crate::display::extractor::IndentedFormatter;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds a `metadata: ...` detail line.
pub struct MetadataExtractor;

impl TreeExtractor for MetadataExtractor {
    fn write_details(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let (indent, f) = f.parts();
        write!(f, "{indent}metadata: ")?;
        array.metadata_fmt(f)?;
        writeln!(f)
    }
}
