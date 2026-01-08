// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{self};

use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::display::DisplayOptions;

pub(crate) struct TreeNoMetadataDisplayWrapper(pub(crate) ArrayRef);

impl fmt::Display for TreeNoMetadataDisplayWrapper {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut array_fmt = TreeFormatter {
            fmt,
            indent: "".to_string(),
        };
        array_fmt.format("root", self.0.clone())
    }
}

pub struct TreeFormatter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    indent: String,
}

impl<'a, 'b: 'a> TreeFormatter<'a, 'b> {
    fn format(&mut self, name: &str, array: ArrayRef) -> fmt::Result {
        writeln!(
            self,
            "{}: {}",
            name,
            array.display_as(DisplayOptions::MetadataOnly),
        )?;

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
