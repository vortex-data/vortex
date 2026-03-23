// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use crate::DynArray;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds a `metadata: ...` detail line.
pub struct MetadataExtractor;

impl TreeExtractor for MetadataExtractor {
    fn detail_lines(&self, array: &dyn DynArray, _ctx: &TreeContext) -> Vec<String> {
        // Capture the metadata_fmt output
        let mut buf = String::new();
        // metadata_fmt writes directly to a Formatter, so we use a helper wrapper
        struct FmtCapture<'a>(&'a dyn DynArray);
        impl fmt::Display for FmtCapture<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.metadata_fmt(f)
            }
        }
        let _ = write!(&mut buf, "{}", FmtCapture(array));
        vec![format!("metadata: {buf}")]
    }
}
