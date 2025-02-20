use std::fmt;

pub struct IndentFormatter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    indent: String,
}

impl<'a, 'b: 'a> IndentFormatter<'a, 'b> {
    pub fn new(fmt: &'a mut fmt::Formatter<'b>, indent: String) -> Self {
        IndentFormatter { fmt, indent }
    }

    pub fn indent<F>(&mut self, indented: F) -> fmt::Result
    where
        F: FnOnce(&mut IndentFormatter) -> fmt::Result,
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
