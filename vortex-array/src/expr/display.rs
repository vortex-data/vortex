// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use crate::expr::Expression;

pub enum DisplayFormat {
    Compact,
    Tree,
}

pub struct DisplayTreeExpr<'a>(pub &'a Expression);

impl Display for DisplayTreeExpr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        pub use termtree::Tree;
        fn make_tree(expr: &Expression) -> Result<Tree<String>, std::fmt::Error> {
            let node_name = format!("{}", ExpressionDebug(expr));

            // Get child names for display purposes
            let child_names = (0..expr.children().len()).map(|i| expr.child_name(i));
            let children = expr.children();

            let child_trees: Result<Vec<Tree<String>>, _> = children
                .iter()
                .zip(child_names)
                .map(|(child, name)| {
                    let child_tree = make_tree(child)?;
                    Ok(Tree::new(format!("{}: {}", name, child_tree.root))
                        .with_leaves(child_tree.leaves))
                })
                .collect();

            Ok(Tree::new(node_name).with_leaves(child_trees?))
        }

        write!(f, "{}", make_tree(self.0)?)
    }
}

struct ExpressionDebug<'a>(&'a Expression);
impl Display for ExpressionDebug<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Special-case when expression has no data to avoid trailing space.
        if self.0.data().is::<()>() {
            return write!(f, "{}", self.0.id().as_ref());
        }
        write!(f, "{} ", self.0.id().as_ref())?;
        self.0.vtable().as_dyn().fmt_data(self.0.data().as_ref(), f)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::compute::{BetweenOptions, StrictComparison};
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::{and, eq, gt};
    use crate::expr::exprs::cast::cast;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::not::not;
    use crate::expr::exprs::pack::pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::{select, select_exclude};

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
    fn test_display_tree() {
        use insta::assert_snapshot;

        let root_expr = root();
        assert_snapshot!(root_expr.display_tree().to_string(), @"vortex.root");

        let lit_expr = lit(42);
        assert_snapshot!(lit_expr.display_tree().to_string(), @"vortex.literal 42i32");

        let get_item_expr = get_item("my_field", root());
        assert_snapshot!(get_item_expr.display_tree().to_string(), @r#"
        vortex.get_item "my_field"
        └── input: vortex.root
        "#);

        let binary_expr = gt(get_item("x", root()), lit(10));
        assert_snapshot!(binary_expr.display_tree().to_string(), @r#"
        vortex.binary >
        ├── lhs: vortex.get_item "x"
        │   └── input: vortex.root
        └── rhs: vortex.literal 10i32
        "#);

        let complex_binary = and(
            eq(get_item("name", root()), lit("alice")),
            gt(get_item("age", root()), lit(18)),
        );
        assert_snapshot!(complex_binary.display_tree().to_string(), @r#"
        vortex.binary and
        ├── lhs: vortex.binary =
        │   ├── lhs: vortex.get_item "name"
        │   │   └── input: vortex.root
        │   └── rhs: vortex.literal "alice"
        └── rhs: vortex.binary >
            ├── lhs: vortex.get_item "age"
            │   └── input: vortex.root
            └── rhs: vortex.literal 18i32
        "#);

        let select_expr = select(["name", "age"], root());
        assert_snapshot!(select_expr.display_tree().to_string(), @r"
        vortex.select include={name, age}
        └── child: vortex.root
        ");

        let select_exclude_expr = select_exclude(["internal_id", "metadata"], root());
        assert_snapshot!(select_exclude_expr.display_tree().to_string(), @r"
        vortex.select exclude={internal_id, metadata}
        └── child: vortex.root
        ");

        let cast_expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_snapshot!(cast_expr.display_tree().to_string(), @r#"
        vortex.cast i64
        └── input: vortex.get_item "value"
            └── input: vortex.root
        "#);

        let not_expr = not(eq(get_item("active", root()), lit(true)));
        assert_snapshot!(not_expr.display_tree().to_string(), @r#"
        vortex.not
        └── input: vortex.binary =
            ├── lhs: vortex.get_item "active"
            │   └── input: vortex.root
            └── rhs: vortex.literal true
        "#);

        let between_expr = between(
            get_item("score", root()),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_snapshot!(between_expr.display_tree().to_string(), @r#"
        vortex.between BetweenOptions { lower_strict: NonStrict, upper_strict: NonStrict }
        ├── array: vortex.get_item "score"
        │   └── input: vortex.root
        ├── lower: vortex.literal 0i32
        └── upper: vortex.literal 100i32
        "#);

        // Test nested expression
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
        assert_snapshot!(nested_expr.display_tree().to_string(), @r#"
        vortex.select include={result}
        └── child: vortex.cast bool
            └── input: vortex.between BetweenOptions { lower_strict: Strict, upper_strict: NonStrict }
                ├── array: vortex.get_item "score"
                │   └── input: vortex.root
                ├── lower: vortex.literal 50i32
                └── upper: vortex.literal 100i32
        "#);

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
        assert_snapshot!(select_from_pack_expr.display_tree().to_string(), @r#"
        vortex.select include={fizz, buzz}
        └── child: vortex.pack PackOptions { names: FieldNames([FieldName("fizz"), FieldName("bar"), FieldName("buzz")]), nullability: Nullable }
            ├── fizz: vortex.root
            ├── bar: vortex.literal 5i32
            └── buzz: vortex.binary =
                ├── lhs: vortex.literal 42i32
                └── rhs: vortex.get_item "answer"
                    └── input: vortex.root
        "#);
    }
}
