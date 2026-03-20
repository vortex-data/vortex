// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;
use crate::display::extractors::BufferExtractor;
use crate::display::extractors::MetadataExtractor;
use crate::display::extractors::NbytesExtractor;
use crate::display::extractors::StatsExtractor;
use crate::display::tree_display::TreeDisplay;

/// Backward-compatible wrapper that maps the old boolean flags to the new extractor-based system.
#[derive(Clone)]
pub(crate) struct TreeDisplayWrapper {
    pub(crate) array: ArrayRef,
    pub(crate) buffers: bool,
    pub(crate) metadata: bool,
    pub(crate) stats: bool,
}

impl fmt::Display for TreeDisplayWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extractors: [(bool, Box<dyn super::TreeExtractor>); 4] = [
            (self.stats, Box::new(NbytesExtractor)),
            (self.stats, Box::new(StatsExtractor)),
            (self.metadata, Box::new(MetadataExtractor)),
            (
                self.buffers,
                Box::new(BufferExtractor {
                    show_percent: self.stats,
                }),
            ),
        ];
        let mut display = TreeDisplay::new(self.array.clone());
        for (enabled, extractor) in extractors {
            if enabled {
                display = display.with_boxed(extractor);
            }
        }
        write!(f, "{display}")
    }
}
