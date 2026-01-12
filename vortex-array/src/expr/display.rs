// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Deref;

use crate::expr::Expression;
use crate::expr::ScalarFn;

pub enum DisplayFormat {
    Compact,
    Tree,
}

pub struct DisplayTreeExpr<'a>(pub &'a Expression);

impl Display for DisplayTreeExpr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        pub use termtree::Tree;
        fn make_tree(expr: &Expression) -> Result<Tree<String>, std::fmt::Error> {
            let scalar_fn: &ScalarFn = expr.deref();
            let node_name = format!("{}", scalar_fn);

            // Get child names for display purposes
            let child_names = (0..expr.children().len()).map(|i| expr.signature().child_name(i));
            let children = expr.children();

            let child_trees: Result<Vec<Tree<String>>, std::fmt::Error> = children
                .iter()
                .zip(child_names)
                .map(|(child, name)| {
                    let child_tree = make_tree(child)?;
                    Ok::<Tree<String>, std::fmt::Error>(
                        Tree::new(format!("{}: {}", name, child_tree.root))
                            .with_leaves(child_tree.leaves),
                    )
                })
                .collect();

            Ok(Tree::new(node_name).with_leaves(child_trees?))
        }

        write!(f, "{}", make_tree(self.0)?)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::compute::BetweenOptions;
    use crate::compute::StrictComparison;
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::cast::cast;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::not::not;
    use crate::expr::exprs::pack::pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::select;
    use crate::expr::exprs::select::select_exclude;

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
