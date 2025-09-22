// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Display, Formatter};

use itertools::Itertools;

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
        pub use termtree::Tree;

        fn make_tree(expr: &dyn Operator) -> Result<Tree<String>, fmt::Error> {
            let node_name = TreeNodeDisplay(expr).to_string();
            let child_trees: Vec<_> = expr
                .children()
                .iter()
                .map(|child| make_tree(child.as_ref()))
                .try_collect()?;
            Ok(Tree::new(node_name).with_leaves(child_trees))
        }

        write!(f, "{}", make_tree(self.0)?)
    }
}

struct TreeNodeDisplay<'a>(&'a dyn Operator);

impl<'a> Display for TreeNodeDisplay<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormat::Tree, f)
    }
}
