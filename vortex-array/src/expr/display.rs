// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tree rendering shared by bound and unbound expressions.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_utils::tree::TreeDisplayAdapter;
use vortex_utils::tree::write_branch_tree;

use crate::expr::BoundExpression;
use crate::scalar_fn::ChildName;

/// A node of an expression tree that [`DisplayTreeExpr`] can render.
pub trait DisplayTreeNode: Sized {
    /// The children of this node, in argument order.
    fn tree_children(&self) -> &[Self];

    /// The display name of the child at `index`.
    fn tree_child_name(&self, index: usize) -> ChildName;

    /// Write this node's own label.
    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

/// Tree-display label for the scope root.
pub const ROOT_DISPLAY: &str = "vortex.root()";

impl DisplayTreeNode for BoundExpression {
    fn tree_children(&self) -> &[Self] {
        BoundExpression::children(self)
    }

    fn tree_child_name(&self, index: usize) -> ChildName {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.signature().child_name(index),
            BoundExpression::Root { .. } => unreachable!("the scope root has no children"),
        }
    }

    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => Display::fmt(scalar_fn, f),
            BoundExpression::Root { .. } => write!(f, "{ROOT_DISPLAY}"),
        }
    }
}

/// Renders an expression tree with one node per line.
pub struct DisplayTreeExpr<'a, T: ?Sized>(pub &'a T);

impl<T: DisplayTreeNode> TreeDisplayAdapter for DisplayTreeExpr<'_, T> {
    type Context = ();
    type Node = T;

    fn write_node(
        &self,
        node: &Self::Node,
        _context: &Self::Context,
        formatter: &mut Formatter<'_>,
    ) -> fmt::Result {
        node.fmt_tree_node(formatter)
    }

    fn visit_children(
        &self,
        node: &Self::Node,
        visit: &mut dyn FnMut(&str, &Self::Node, bool) -> fmt::Result,
    ) -> fmt::Result {
        let children = node.tree_children();
        for (index, child) in children.iter().enumerate() {
            let child_name = node.tree_child_name(index);
            visit(child_name.as_ref(), child, index + 1 == children.len())?;
        }
        Ok(())
    }
}

impl<T: DisplayTreeNode> Display for DisplayTreeExpr<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write_branch_tree(self, self.0, &mut (), f)
    }
}
