// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::io;
use std::io::Write;

pub struct IndentedWriter<W: Write> {
    write: W,
    indent: String,
}

impl<W: Write> IndentedWriter<W> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            indent: String::new(),
        }
    }

    /// # Errors
    ///
    /// Will return Err if writing to the underlying writer fails.
    pub fn indent<F>(&mut self, indented: F) -> io::Result<()>
    where
        F: FnOnce(&mut IndentedWriter<W>) -> io::Result<()>,
    {
        let original_ident = self.indent.clone();
        self.indent += "    ";
        let res = indented(self);
        self.indent = original_ident;
        res
    }

    /// # Errors
    ///
    /// Will return Err if writing to the underlying writer fails.
    pub fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        write!(self.write, "{}{}", self.indent, fmt)
    }
}
