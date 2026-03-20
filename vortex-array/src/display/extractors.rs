// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use humansize::DECIMAL;
use humansize::format_size;

use crate::DynArray;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// Display wrapper for array statistics in compact format.
///
/// Produces output like ` [nulls=3, min=5, max=100]` (with leading space).
pub(crate) struct StatsDisplay<'a>(pub(crate) &'a dyn DynArray);

impl fmt::Display for StatsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.0.statistics();
        let mut first = true;

        let mut sep = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
            if first {
                first = false;
                f.write_str(" [")
            } else {
                f.write_str(", ")
            }
        };

        // Null count or validity fallback
        if let Some(nc) = stats.get(Stat::NullCount) {
            if let Ok(n) = usize::try_from(&nc.clone().into_inner()) {
                sep(f)?;
                write!(f, "nulls={}", n)?;
            } else {
                sep(f)?;
                write!(f, "nulls={}", nc)?;
            }
        } else if self.0.dtype().is_nullable() {
            match self.0.all_valid() {
                Ok(true) => {
                    sep(f)?;
                    f.write_str("all_valid")?;
                }
                Ok(false) => {
                    if self.0.all_invalid().unwrap_or(false) {
                        sep(f)?;
                        f.write_str("all_invalid")?;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to check validity: {e}");
                    sep(f)?;
                    f.write_str("validity_failed")?;
                }
            }
        }

        // NaN count (only if > 0)
        if let Some(nan) = stats.get(Stat::NaNCount)
            && let Ok(n) = usize::try_from(&nan.into_inner())
            && n > 0
        {
            sep(f)?;
            write!(f, "nan={}", n)?;
        }

        // Min/Max
        if let Some(min) = stats.get(Stat::Min) {
            sep(f)?;
            write!(f, "min={}", min)?;
        }
        if let Some(max) = stats.get(Stat::Max) {
            sep(f)?;
            write!(f, "max={}", max)?;
        }

        // Sum
        if let Some(sum) = stats.get(Stat::Sum) {
            sep(f)?;
            write!(f, "sum={}", sum)?;
        }

        // Boolean flags (compact)
        if let Some(c) = stats.get(Stat::IsConstant)
            && bool::try_from(&c.into_inner()).unwrap_or(false)
        {
            sep(f)?;
            f.write_str("const")?;
        }
        if let Some(s) = stats.get(Stat::IsStrictSorted) {
            if bool::try_from(&s.into_inner()).unwrap_or(false) {
                sep(f)?;
                f.write_str("strict")?;
            }
        } else if let Some(s) = stats.get(Stat::IsSorted)
            && bool::try_from(&s.into_inner()).unwrap_or(false)
        {
            sep(f)?;
            f.write_str("sorted")?;
        }

        // Close bracket if we wrote anything
        if !first {
            f.write_char(']')?;
        }

        Ok(())
    }
}

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

/// Extractor that adds stats annotations (e.g. `[nulls=3, min=5]`) to the header line.
pub struct StatsExtractor;

impl TreeExtractor for StatsExtractor {
    fn header_annotations(&self, array: &dyn DynArray, _ctx: &TreeContext) -> Vec<String> {
        let s = StatsDisplay(array).to_string();
        let trimmed = s.trim_start();
        if trimmed.is_empty() {
            vec![]
        } else {
            vec![trimmed.to_string()]
        }
    }
}

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
