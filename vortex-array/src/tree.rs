use std::fmt::{self};

use humansize::{format_size, DECIMAL};
use serde::ser::Error;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult};

use crate::arrays::ChunkedEncoding;
use crate::visitor::ArrayVisitor;
use crate::vtable::EncodingVTable;
use crate::Array;

impl Array {
    pub fn tree_display(&self) -> impl fmt::Display + use<'_> {
        TreeDisplayWrapper(self)
    }
}

pub struct TreeDisplayWrapper<'a>(pub &'a Array);

impl fmt::Display for TreeDisplayWrapper<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let array = self.0;
        let mut array_fmt = TreeFormatter::new(f, "".to_string());
        array_fmt
            .visit_child("root", array)
            .map_err(fmt::Error::custom)
    }
}

pub struct TreeFormatter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    indent: String,
    total_size: Option<usize>,
}

/// TODO(ngates): I think we want to go back to the old explicit style. It gives arrays more
///  control over how their metadata etc is displayed.
impl<'a, 'b: 'a> ArrayVisitor for TreeFormatter<'a, 'b> {
    fn visit_child(&mut self, name: &str, array: &Array) -> VortexResult<()> {
        let nbytes = array.nbytes();
        let total_size = self.total_size.unwrap_or(nbytes);
        writeln!(
            self.fmt,
            "{}{}: {} nbytes={} ({:.2}%)",
            self.indent,
            name,
            array,
            format_size(nbytes, DECIMAL),
            100f64 * nbytes as f64 / total_size as f64
        )?;
        self.indent(|i| {
            write!(i.fmt, "{}metadata: ", i.indent)?;
            array.vtable().display_metadata(array, i.fmt)?;
            writeln!(i.fmt)
        })?;

        let old_total_size = self.total_size;
        if array.is_encoding(ChunkedEncoding.id()) {
            // Clear the total size so each chunk is treated as a new root.
            self.total_size = None
        } else {
            self.total_size = Some(total_size);
        }

        self.indent(|i| array.vtable().accept(array, i).map_err(fmt::Error::custom))
            .map_err(VortexError::from)?;

        self.total_size = old_total_size;
        Ok(())
    }

    fn visit_buffer(&mut self, buffer: &ByteBuffer) -> VortexResult<()> {
        Ok(writeln!(
            self.fmt,
            "{}buffer (align={}): {}",
            self.indent,
            buffer.alignment(),
            format_size(buffer.len(), DECIMAL)
        )?)
    }
}

impl<'a, 'b: 'a> TreeFormatter<'a, 'b> {
    pub fn new(fmt: &'a mut fmt::Formatter<'b>, indent: String) -> Self {
        TreeFormatter {
            fmt,
            indent,
            total_size: None,
        }
    }

    pub fn indent<F>(&mut self, indented: F) -> fmt::Result
    where
        F: FnOnce(&mut TreeFormatter) -> fmt::Result,
    {
        let original_ident = self.indent.clone();
        self.indent += "  ";
        let res = indented(self);
        self.indent = original_ident;
        res
    }

    pub fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> fmt::Result {
        write!(self.fmt, "{}{}", self.indent, fmt)
    }
}
