// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use humansize::DECIMAL;
use humansize::format_size;

use crate::DynArray;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds buffer detail lines.
pub struct BufferExtractor {
    /// Whether to show buffer-level percentage of parent nbytes.
    pub show_percent: bool,
}

impl TreeExtractor for BufferExtractor {
    fn detail_lines(&self, array: &dyn DynArray, _ctx: &TreeContext) -> Vec<String> {
        let nbytes = array.nbytes();
        let mut lines = Vec::new();
        for (name, buffer) in array.named_buffers() {
            let loc = if buffer.is_on_device() {
                "device"
            } else if buffer.is_on_host() {
                "host"
            } else {
                "location-unknown"
            };
            let align = if buffer.is_on_host() {
                buffer.as_host().alignment().to_string()
            } else {
                String::new()
            };

            if self.show_percent {
                let buffer_percent = if nbytes == 0 {
                    0.0
                } else {
                    100_f64 * buffer.len() as f64 / nbytes as f64
                };
                lines.push(format!(
                    "buffer: {} {loc} {} (align={}) ({:.2}%)",
                    name,
                    format_size(buffer.len(), DECIMAL),
                    align,
                    buffer_percent,
                ));
            } else {
                lines.push(format!(
                    "buffer: {} {loc} {} (align={})",
                    name,
                    format_size(buffer.len(), DECIMAL),
                    align,
                ));
            }
        }
        lines
    }
}
