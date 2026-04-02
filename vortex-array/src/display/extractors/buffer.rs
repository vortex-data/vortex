// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use humansize::DECIMAL;
use humansize::format_size;

use crate::ArrayRef;
use crate::display::extractor::IndentedFormatter;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;

/// Extractor that adds buffer detail lines.
pub struct BufferExtractor {
    /// Whether to show buffer-level percentage of parent nbytes.
    pub show_percent: bool,
}

impl TreeExtractor for BufferExtractor {
    fn write_details(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let (indent, f) = f.parts();
        let nbytes = array.nbytes();
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
                writeln!(
                    f,
                    "{indent}buffer: {} {loc} {} (align={}) ({:.2}%)",
                    name,
                    format_size(buffer.len(), DECIMAL),
                    align,
                    buffer_percent,
                )?;
            } else {
                writeln!(
                    f,
                    "{indent}buffer: {} {loc} {} (align={})",
                    name,
                    format_size(buffer.len(), DECIMAL),
                    align,
                )?;
            }
        }
        Ok(())
    }
}
