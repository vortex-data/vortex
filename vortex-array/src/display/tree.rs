// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use humansize::DECIMAL;
use humansize::format_size;
use vortex_error::VortexExpect as _;

use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::DynArray;
use crate::arrays::ChunkedVTable;
use crate::display::DisplayOptions;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// Display wrapper for array statistics in compact format.
struct StatsDisplay<'a>(&'a dyn DynArray);

impl fmt::Display for StatsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.0.statistics();
        let mut first = true;

        // Helper to write separator
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

#[derive(Clone)]
pub(crate) struct TreeDisplayWrapper {
    pub(crate) array: ArrayRef,
    pub(crate) buffers: bool,
    pub(crate) metadata: bool,
    pub(crate) stats: bool,
}

impl fmt::Display for TreeDisplayWrapper {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let TreeDisplayWrapper {
            array,
            buffers,
            metadata,
            stats,
        } = self.clone();
        let mut array_fmt = TreeFormatter {
            fmt,
            indent: "".to_string(),
            ancestor_sizes: Vec::new(),
            buffers,
            metadata,
            stats,
        };
        array_fmt.format("root", array)
    }
}

pub struct TreeFormatter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    indent: String,
    ancestor_sizes: Vec<Option<u64>>,
    buffers: bool,
    metadata: bool,
    stats: bool,
}

impl<'a, 'b: 'a> TreeFormatter<'a, 'b> {
    fn format(&mut self, name: &str, array: ArrayRef) -> fmt::Result {
        if self.stats {
            let nbytes = array.nbytes();
            let total_size = self
                .ancestor_sizes
                .last()
                .cloned()
                .flatten()
                .unwrap_or(nbytes);

            self.ancestor_sizes.push(if array.is::<ChunkedVTable>() {
                // Treat each chunk as a new root
                None
            } else {
                // Children will present themselves as a percentage of our size.
                Some(nbytes)
            });
            let percent = if total_size == 0 {
                0.0
            } else {
                100_f64 * nbytes as f64 / total_size as f64
            };

            writeln!(
                self,
                "{}: {} nbytes={} ({:.2}%){}",
                name,
                array.display_as(DisplayOptions::MetadataOnly),
                format_size(nbytes, DECIMAL),
                percent,
                StatsDisplay(array.as_ref()),
            )?;
        } else {
            writeln!(
                self,
                "{}: {}",
                name,
                array.display_as(DisplayOptions::MetadataOnly)
            )?;
        }

        self.indent(|i| {
            if i.metadata {
                write!(i, "metadata: ")?;
                array.metadata_fmt(i.fmt)?;
                writeln!(i.fmt)?;
            }

            if i.buffers {
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
                        "".to_string()
                    };

                    if i.stats {
                        let buffer_percent = if nbytes == 0 {
                            0.0
                        } else {
                            100_f64 * buffer.len() as f64 / nbytes as f64
                        };
                        writeln!(
                            i,
                            "buffer: {} {loc} {} (align={}) ({:.2}%)",
                            name,
                            format_size(buffer.len(), DECIMAL),
                            align,
                            buffer_percent
                        )?;
                    } else {
                        writeln!(
                            i,
                            "buffer: {} {loc} {} (align={})",
                            name,
                            format_size(buffer.len(), DECIMAL),
                            align,
                        )?;
                    }
                }
            }

            Ok(())
        })?;

        self.indent(|i| {
            for (name, child) in array
                .children_names()
                .into_iter()
                .zip(array.children().into_iter())
            {
                i.format(&name, child)?;
            }
            Ok(())
        })?;

        if self.stats {
            let _ = self
                .ancestor_sizes
                .pop()
                .vortex_expect("pushes and pops are matched");
        }

        Ok(())
    }

    fn indent<F>(&mut self, indented: F) -> fmt::Result
    where
        F: FnOnce(&mut TreeFormatter) -> fmt::Result,
    {
        let original_ident = self.indent.clone();
        self.indent += "  ";
        let res = indented(self);
        self.indent = original_ident;
        res
    }

    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> fmt::Result {
        write!(self.fmt, "{}{}", self.indent, fmt)
    }
}
