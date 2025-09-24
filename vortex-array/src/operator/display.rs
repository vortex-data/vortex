// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Display, Formatter};

use crate::operator::Operator;

impl dyn Operator + '_ {
    pub fn display_tree(&self) -> impl Display {
        self
    }
}

pub enum DisplayFormat {
    Compact,
    Tree,
}

impl Display for dyn Operator + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fmt_all())
    }
}

pub struct TreeNodeDisplay<'a, T: Operator + ?Sized>(pub &'a T);

impl<'a, T: Operator + ?Sized> Display for TreeNodeDisplay<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormat::Tree, f)
    }
}
