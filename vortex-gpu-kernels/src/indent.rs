// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::{fmt, io};

pub struct IndentedWriter<W: Write> {
    write: W,
    indent: String,
}

impl<W: Write> IndentedWriter<W> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            indent: "".to_string(),
        }
    }

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

    pub fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        write!(self.write, "{}{}", self.indent, fmt)
    }
}
