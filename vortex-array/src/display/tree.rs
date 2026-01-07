// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use humansize::DECIMAL;
use humansize::format_size;

use crate::Array;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::arrays::ChunkedVTable;
use crate::display::DisplayOptions;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// Display wrapper for array statistics in compact format.
struct StatsDisplay<'a>(&'a dyn Array);

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
            if self.0.all_valid() {
                sep(f)?;
                f.write_str("all_valid")?;
            } else if self.0.all_invalid() {
                sep(f)?;
                f.write_str("all_invalid")?;
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

pub(crate) struct TreeDisplayWrapper(pub(crate) ArrayRef);

impl fmt::Display for TreeDisplayWrapper {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut array_fmt = TreeFormatter {
            fmt,
            indent: "".to_string(),
            total_size: None,
        };
        array_fmt.format("root", self.0.clone())
    }
}

pub struct TreeFormatter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    indent: String,
    total_size: Option<u64>,
}

impl<'a, 'b: 'a> TreeFormatter<'a, 'b> {
    fn format(&mut self, name: &str, array: ArrayRef) -> fmt::Result {
        let nbytes = array.nbytes();
        let total_size = self.total_size.unwrap_or(nbytes);
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

        self.indent(|i| {
            write!(i, "metadata: ")?;
            array.metadata_fmt(i.fmt)?;
            writeln!(i.fmt)?;

            for buffer in array.buffers() {
                let buffer_percent = if nbytes == 0 {
                    0.0
                } else {
                    100_f64 * buffer.len() as f64 / nbytes as f64
                };
                writeln!(
                    i,
                    "buffer (align={}): {} ({:.2}%)",
                    buffer.alignment(),
                    format_size(buffer.len(), DECIMAL),
                    buffer_percent
                )?;
            }

            Ok(())
        })?;

        let old_total_size = self.total_size;
        if array.is::<ChunkedVTable>() {
            // Clear the total size so each chunk is treated as a new root.
            self.total_size = None
        } else {
            self.total_size = Some(nbytes);
        }

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

        self.total_size = old_total_size;
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
