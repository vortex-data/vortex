// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_utils::tree::TreeDisplayAdapter;
use vortex_utils::tree::write_branch_tree;

use crate::expr::BoundExpression;
use crate::expr::BoundKind;
use crate::expr::Expression;
use crate::expr::root;
use crate::scalar_fn::ChildName;

pub enum DisplayFormat {
    Compact,
    Tree,
}

/// Read-only expression-tree interface used by scalar functions for SQL-style formatting.
///
/// Both [`Expression`] and [`BoundExpression`] implement this interface, allowing scalar
/// functions to format either representation without converting between them.
pub trait ExprDisplay: Display {
    /// Return the child at `index`.
    fn display_child(&self, index: usize) -> &dyn ExprDisplay;

    /// Return the number of children in this node.
    fn display_children_count(&self) -> usize;
}

impl ExprDisplay for Expression {
    fn display_child(&self, index: usize) -> &dyn ExprDisplay {
        Expression::child(self, index)
    }

    fn display_children_count(&self) -> usize {
        self.children().len()
    }
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

impl DisplayTreeNode for Expression {
    fn tree_children(&self) -> &[Self] {
        Expression::children(self).as_slice()
    }

    fn tree_child_name(&self, index: usize) -> ChildName {
        self.scalar_fn().signature().child_name(index)
    }

    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.scalar_fn(), f)
    }
}

impl DisplayTreeNode for BoundExpression {
    fn tree_children(&self) -> &[Self] {
        BoundExpression::children(self)
    }

    fn tree_child_name(&self, index: usize) -> ChildName {
        match self.kind() {
            BoundKind::Scalar { scalar_fn, .. } => scalar_fn.signature().child_name(index),
            BoundKind::Root => unreachable!("the scope root has no children"),
        }
    }

    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.kind() {
            BoundKind::Scalar { scalar_fn, .. } => Display::fmt(scalar_fn, f),
            BoundKind::Root => Display::fmt(root().scalar_fn(), f),
        }
    }
}

pub struct DisplayTreeExpr<'a, T: ?Sized = Expression>(pub &'a T);

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
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::cast;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;

    #[test]
    fn tree_display_getitem() {
        let expr = get_item("x", root());
        println!("{}", expr.display_tree());
    }

    #[test]
    fn tree_display_binary() {
        let expr = gt(get_item("x", root()), lit(5));
        println!("{}", expr.display_tree());
    }

    #[test]
    fn test_child_names_debug() {
        // Simple test to debug child names display
        let binary_expr = gt(get_item("x", root()), lit(10));
        println!("Binary expr tree:\n{}", binary_expr.display_tree());

        let between_expr = between(
            get_item("score", root()),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        println!("Between expr tree:\n{}", between_expr.display_tree());
    }

    #[test]
    fn test_display_tree_root() {
        use insta::assert_snapshot;
        let root_expr = root();
        assert_snapshot!(root_expr.display_tree().to_string(), @"vortex.root()");
    }

    #[test]
    fn test_display_tree_literal() {
        use insta::assert_snapshot;
        let lit_expr = lit(42);
        assert_snapshot!(lit_expr.display_tree().to_string(), @"vortex.literal(42i32)");
    }

    #[test]
    fn test_display_tree_get_item() {
        use insta::assert_snapshot;
        let get_item_expr = get_item("my_field", root());
        assert_snapshot!(get_item_expr.display_tree().to_string(), @r"
        vortex.get_item(my_field)
        └── input: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_binary() {
        use insta::assert_snapshot;
        let binary_expr = gt(get_item("x", root()), lit(10));
        assert_snapshot!(binary_expr.display_tree().to_string(), @r"
        vortex.binary(>)
        ├── lhs: vortex.get_item(x)
        │   └── input: vortex.root()
        └── rhs: vortex.literal(10i32)
        ");
    }

    #[test]
    fn test_display_tree_complex_binary() {
        use insta::assert_snapshot;
        let complex_binary = and(
            eq(get_item("name", root()), lit("alice")),
            gt(get_item("age", root()), lit(18)),
        );
        assert_snapshot!(complex_binary.display_tree().to_string(), @r#"
        vortex.binary(and)
        ├── lhs: vortex.binary(=)
        │   ├── lhs: vortex.get_item(name)
        │   │   └── input: vortex.root()
        │   └── rhs: vortex.literal("alice")
        └── rhs: vortex.binary(>)
            ├── lhs: vortex.get_item(age)
            │   └── input: vortex.root()
            └── rhs: vortex.literal(18i32)
        "#);
    }

    #[test]
    fn test_display_tree_select() {
        use insta::assert_snapshot;
        let select_expr = select(["name", "age"], root());
        assert_snapshot!(select_expr.display_tree().to_string(), @r"
        vortex.select({name, age})
        └── child: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_select_exclude() {
        use insta::assert_snapshot;
        let select_exclude_expr = select_exclude(["internal_id", "metadata"], root());
        assert_snapshot!(select_exclude_expr.display_tree().to_string(), @r"
        vortex.select(~{internal_id, metadata})
        └── child: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_cast() {
        use insta::assert_snapshot;
        let cast_expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_snapshot!(cast_expr.display_tree().to_string(), @r"
        vortex.cast(i64)
        └── input: vortex.get_item(value)
            └── input: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_not() {
        use insta::assert_snapshot;
        let not_expr = not(eq(get_item("active", root()), lit(true)));
        assert_snapshot!(not_expr.display_tree().to_string(), @r"
        vortex.not()
        └── input: vortex.binary(=)
            ├── lhs: vortex.get_item(active)
            │   └── input: vortex.root()
            └── rhs: vortex.literal(true)
        ");
    }

    #[test]
    fn test_display_tree_between() {
        use insta::assert_snapshot;
        let between_expr = between(
            get_item("score", root()),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_snapshot!(between_expr.display_tree().to_string(), @r"
        vortex.between(lower_strict: <=, upper_strict: <=)
        ├── array: vortex.get_item(score)
        │   └── input: vortex.root()
        ├── lower: vortex.literal(0i32)
        └── upper: vortex.literal(100i32)
        ");
    }

    #[test]
    fn test_display_tree_nested() {
        use insta::assert_snapshot;
        let nested_expr = select(
            ["result"],
            cast(
                between(
                    get_item("score", root()),
                    lit(50),
                    lit(100),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::NonStrict,
                    },
                ),
                DType::Bool(Nullability::NonNullable),
            ),
        );
        assert_snapshot!(nested_expr.display_tree().to_string(), @r"
        vortex.select({result})
        └── child: vortex.cast(bool)
            └── input: vortex.between(lower_strict: <, upper_strict: <=)
                ├── array: vortex.get_item(score)
                │   └── input: vortex.root()
                ├── lower: vortex.literal(50i32)
                └── upper: vortex.literal(100i32)
        ");
    }

    #[test]
    fn test_display_tree_pack() {
        use insta::assert_snapshot;
        let select_from_pack_expr = select(
            ["fizz", "buzz"],
            pack(
                [
                    ("fizz", root()),
                    ("bar", lit(5)),
                    ("buzz", eq(lit(42), get_item("answer", root()))),
                ],
                Nullability::Nullable,
            ),
        );
        assert_snapshot!(select_from_pack_expr.display_tree().to_string(), @r"
        vortex.select({fizz, buzz})
        └── child: vortex.pack(names: [fizz, bar, buzz], nullability: Nullable)
            ├── fizz: vortex.root()
            ├── bar: vortex.literal(5i32)
            └── buzz: vortex.binary(=)
                ├── lhs: vortex.literal(42i32)
                └── rhs: vortex.get_item(answer)
                    └── input: vortex.root()
        ");
    }
}
