// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;

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
        match array.metadata() {
            Ok(Some(metadata)) => Debug::fmt(&metadata, f)?,
            Ok(None) => write!(f, "<unsupported>")?,
            Err(err) => write!(f, "<serde error: {err}>")?,
        }
        writeln!(f)
    }
}
