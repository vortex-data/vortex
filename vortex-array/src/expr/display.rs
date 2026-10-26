// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_utils::tree::TreeDisplayAdapter;
use vortex_utils::tree::write_branch_tree;

use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ScalarFnId;

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

/// Format a node as a call of its function's id: `id(child, ..., opts=options)`, leaving the
/// options out when they display as nothing.
///
/// This is the default [`ScalarFnVTable::fmt_sql`](crate::scalar_fn::ScalarFnVTable::fmt_sql),
/// and how every node of an unbound [`Expression`] is formatted.
pub(crate) fn fmt_call(
    id: ScalarFnId,
    options: &dyn Display,
    expr: &dyn ExprDisplay,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    write!(f, "{id}(")?;
    let nchildren = expr.display_children_count();
    for i in 0..nchildren {
        if i > 0 {
            write!(f, ", ")?;
        }
        Display::fmt(expr.display_child(i), f)?;
    }
    let opts = options.to_string();
    if !opts.is_empty() {
        if nchildren > 0 {
            write!(f, ", ")?;
        }
        write!(f, "opts={opts}")?;
    }
    write!(f, ")")
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

/// Tree-display label for the scope root.
const ROOT_DISPLAY: &str = "vortex.root()";

impl DisplayTreeNode for Expression {
    fn tree_children(&self) -> &[Self] {
        Expression::children(self)
    }

    fn tree_child_name(&self, index: usize) -> ChildName {
        match self {
            // The function, and so the children's names, are known once the expression is bound.
            Expression::Scalar { .. } => ChildName::new_arc(format!("arg{index}").into()),
            Expression::Root => unreachable!("the scope root has no children"),
            Expression::Variable(_) => unreachable!("a variable has no children"),
        }
    }

    fn fmt_tree_node(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Scalar { id, options, .. } => write!(f, "{id}({options})"),
            Expression::Root => write!(f, "{ROOT_DISPLAY}"),
            Expression::Variable(name) => write!(f, "@{name}"),
        }
    }
}

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
    use insta::assert_snapshot;
    use vortex_session::VortexSession;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
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

    fn i32_() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// The scope the test expressions are bound against.
    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("my_field", i32_()),
                ("x", i32_()),
                ("name", DType::Utf8(Nullability::NonNullable)),
                ("age", i32_()),
                ("value", i32_()),
                ("active", DType::Bool(Nullability::NonNullable)),
                ("score", i32_()),
                ("answer", i32_()),
                ("internal_id", i32_()),
                ("metadata", DType::Utf8(Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    /// The tree display of `expr` bound against the test scope, with each function's own
    /// formatting and child names.
    fn bound_tree(expr: &Expression) -> String {
        expr.bind(&scope(), &VortexSession::empty())
            .unwrap_or_else(|e| panic!("{expr} binds against the test scope: {e}"))
            .display_tree()
            .to_string()
    }

    #[test]
    fn test_display_tree_unbound() {
        let expr = gt(get_item("x", root()), lit(10));
        assert_snapshot!(expr.display_tree().to_string(), @r"
        vortex.binary(>)
        ├── arg0: vortex.get_item(x)
        │   └── arg0: vortex.root()
        └── arg1: vortex.literal(10i32)
        ");
    }

    #[test]
    fn test_display_tree_root() {
        assert_snapshot!(bound_tree(&root()), @"vortex.root()");
        assert_snapshot!(root().display_tree().to_string(), @"vortex.root()");
    }

    #[test]
    fn test_display_tree_literal() {
        assert_snapshot!(bound_tree(&lit(42)), @"vortex.literal(42i32)");
        assert_snapshot!(lit(42).display_tree().to_string(), @"vortex.literal(42i32)");
    }

    #[test]
    fn test_display_tree_get_item() {
        let get_item_expr = get_item("my_field", root());
        assert_snapshot!(bound_tree(&get_item_expr), @r"
        vortex.get_item(my_field)
        └── input: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_binary() {
        let binary_expr = gt(get_item("x", root()), lit(10));
        assert_snapshot!(bound_tree(&binary_expr), @r"
        vortex.binary(>)
        ├── lhs: vortex.get_item(x)
        │   └── input: vortex.root()
        └── rhs: vortex.literal(10i32)
        ");
    }

    #[test]
    fn test_display_tree_complex_binary() {
        let complex_binary = and(
            eq(get_item("name", root()), lit("alice")),
            gt(get_item("age", root()), lit(18)),
        );
        assert_snapshot!(bound_tree(&complex_binary), @r#"
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
        let select_expr = select(["name", "age"], root());
        assert_snapshot!(bound_tree(&select_expr), @r"
        vortex.select({name, age})
        └── child: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_select_exclude() {
        let select_exclude_expr = select_exclude(["internal_id", "metadata"], root());
        assert_snapshot!(bound_tree(&select_exclude_expr), @r"
        vortex.select(~{internal_id, metadata})
        └── child: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_cast() {
        let cast_expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_snapshot!(bound_tree(&cast_expr), @r"
        vortex.cast(i64)
        └── input: vortex.get_item(value)
            └── input: vortex.root()
        ");
    }

    #[test]
    fn test_display_tree_not() {
        let not_expr = not(eq(get_item("active", root()), lit(true)));
        assert_snapshot!(bound_tree(&not_expr), @r"
        vortex.not()
        └── input: vortex.binary(=)
            ├── lhs: vortex.get_item(active)
            │   └── input: vortex.root()
            └── rhs: vortex.literal(true)
        ");
    }

    #[test]
    fn test_display_tree_between() {
        let between_expr = between(
            get_item("score", root()),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_snapshot!(bound_tree(&between_expr), @r"
        vortex.between(lower_strict: <=, upper_strict: <=)
        ├── array: vortex.get_item(score)
        │   └── input: vortex.root()
        ├── lower: vortex.literal(0i32)
        └── upper: vortex.literal(100i32)
        ");
    }

    #[test]
    fn test_display_tree_nested() {
        let nested_expr = not(cast(
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
        ));
        assert_snapshot!(bound_tree(&nested_expr), @r"
        vortex.not()
        └── input: vortex.cast(bool)
            └── input: vortex.between(lower_strict: <, upper_strict: <=)
                ├── array: vortex.get_item(score)
                │   └── input: vortex.root()
                ├── lower: vortex.literal(50i32)
                └── upper: vortex.literal(100i32)
        ");
    }

    #[test]
    fn test_display_tree_pack() {
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
        assert_snapshot!(bound_tree(&select_from_pack_expr), @r"
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
