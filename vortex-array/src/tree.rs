use std::fmt::{self};

use humansize::{DECIMAL, format_size};

use crate::arrays::ChunkedEncoding;
use crate::nbytes::NBytes;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ArrayVisitor};

impl dyn Array + '_ {
    pub fn tree_display(&self) -> impl fmt::Display {
        TreeDisplayWrapper(self.to_array())
    }
}

struct TreeDisplayWrapper(ArrayRef);

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
    total_size: Option<usize>,
}

impl<'a, 'b: 'a> TreeFormatter<'a, 'b> {
    fn format(&mut self, name: &str, array: ArrayRef) -> fmt::Result {
        let nbytes = array.nbytes();
        let total_size = self.total_size.unwrap_or(nbytes);
        writeln!(
            self,
            "{}: {} nbytes={} ({:.2}%)",
            name,
            array,
            format_size(nbytes, DECIMAL),
            100_f64 * nbytes as f64 / total_size as f64
        )?;

        self.indent(|i| {
            write!(i, "metadata: ")?;
            array.metadata_fmt(i.fmt)?;
            writeln!(i.fmt)?;

            for buffer in array.buffers() {
                writeln!(
                    i,
                    "buffer (align={}): {} ({:.2}%)",
                    buffer.alignment(),
                    format_size(buffer.len(), DECIMAL),
                    100_f64 * buffer.len() as f64 / nbytes as f64
                )?;
            }

            Ok(())
        })?;

        let old_total_size = self.total_size;
        if array.is_encoding(ChunkedEncoding.id()) {
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
