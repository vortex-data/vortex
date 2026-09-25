// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_utils::tree::TreeDisplayAdapter;
use vortex_utils::tree::write_branch_tree;

use crate::expr::BoundExpression;
use crate::scalar_fn::ChildName;

pub enum DisplayFormat {
    Compact,
    Tree,
}

/// Read-only expression-tree interface used by scalar functions for SQL-style formatting.
///
/// Scalar functions use this interface to format bound expressions.
pub trait ExprDisplay: Display {
    /// Return the child at `index`.
    fn display_child(&self, index: usize) -> &dyn ExprDisplay;

    /// Return the number of children in this node.
    fn display_children_count(&self) -> usize;
}

impl ExprDisplay for BoundExpression {
    fn display_child(&self, index: usize) -> &dyn ExprDisplay {
        &self.children()[index]
    }

    fn display_children_count(&self) -> usize {
        self.children().len()
    }
}

trait DisplayTreeNode: Sized {
    fn tree_children(&self) -> &[Self];

    fn tree_child_name(&self, index: usize) -> ChildName;

    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

/// Tree-display label for the scope root.
const ROOT_DISPLAY: &str = "vortex.root()";

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

pub struct DisplayTreeExpr<'a, T: ?Sized = BoundExpression>(pub &'a T);

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

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::lit;
    use crate::expr::root;

    #[test]
    fn display_bound_tree() {
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let root_expr = root(scope.clone());
        insta::assert_snapshot!(root_expr.display_tree(), @"vortex.root()");
        let literal = lit(42i32);
        insta::assert_snapshot!(literal.display_tree(), @"vortex.literal(42i32)");
        let comparison = gt(get_item("x", root(scope)), lit(10i32));
        insta::assert_snapshot!(comparison.display_tree(), @r"
        vortex.binary(>)
        ├── lhs: vortex.get_item(x)
        │   └── input: vortex.root()
        └── rhs: vortex.literal(10i32)
        ");
    }
}
