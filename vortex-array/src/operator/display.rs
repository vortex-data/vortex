// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Display, Formatter};

use crate::operator::Operator;

impl dyn Operator + '_ {
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

pub enum DisplayFormat {
    Compact,
    Tree,
}

// TODO(ngates): this is pretty bad right now, and pipelined operators display poorly.
struct DisplayTreeExpr<'a>(&'a dyn Operator);

impl Display for DisplayTreeExpr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.fmt_all())
    }
}

pub struct TreeNodeDisplay<'a, T: Operator + ?Sized>(pub &'a T);

impl<'a, T: Operator + ?Sized> Display for TreeNodeDisplay<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormat::Tree, f)
    }
}
