// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub enum DisplayFormat {
    Dense,
    #[cfg(feature = "pretty")]
    Tree,
}

/// Configurable display trait for expressions.
pub trait DisplayAs {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result;

    fn child_names(&self) -> Option<Vec<String>> {
        None
    }
}

#[cfg(feature = "pretty")]
pub struct DisplayTreeExpr<'a>(pub &'a dyn crate::VortexExpr);

#[cfg(feature = "pretty")]
impl std::fmt::Display for DisplayTreeExpr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        pub use termtree::Tree;
        fn make_tree(expr: &dyn crate::VortexExpr) -> Result<Tree<String>, std::fmt::Error> {
            let node_name = TreeNodeDisplay(expr).to_string();

            // Get child names for display purposes
            let child_names = DisplayAs::child_names(expr);
            let children = expr.children();

            let child_trees: Result<Vec<Tree<String>>, _> = if let Some(names) = child_names
                && names.len() == children.len()
            {
                children
                    .iter()
                    .zip(names.iter())
                    .map(|(child, name)| {
                        let child_tree = make_tree(child.as_ref())?;
                        Ok(Tree::new(format!("{}: {}", name, child_tree.root))
                            .with_leaves(child_tree.leaves))
                    })
                    .collect()
            } else {
                children
                    .iter()
                    .map(|child| make_tree(child.as_ref()))
                    .collect()
            };

            Ok(Tree::new(node_name).with_leaves(child_trees?))
        }

        write!(f, "{}", make_tree(self.0)?)
    }
}

#[cfg(feature = "pretty")]
struct TreeNodeDisplay<'a>(&'a dyn crate::VortexExpr);

#[cfg(feature = "pretty")]
impl<'a> std::fmt::Display for TreeNodeDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(DisplayFormat::Tree, f)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{and, between, cast, eq, get_item, gt, lit, not, root, select};

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

    #[cfg(feature = "pretty")]
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

    #[cfg(feature = "pretty")]
    #[test]
    fn test_display_tree() {
        use insta::assert_snapshot;

        use crate::{pack, select_exclude};
        let root_expr = root();
        assert_snapshot!(root_expr.display_tree().to_string(), @"RootExpr");

        let lit_expr = lit(42);
        assert_snapshot!(lit_expr.display_tree().to_string(), @"LiteralExpr(value: 42i32, dtype: i32)");

        let get_item_expr = get_item("my_field", root());
        assert_snapshot!(get_item_expr.display_tree().to_string(), @r"
        GetItemExpr(field = my_field)
        └── RootExpr
        ");

        let binary_expr = gt(get_item("x", root()), lit(10));
        assert_snapshot!(binary_expr.display_tree().to_string(), @r"
        BinaryExpr(>)
        ├── lhs: GetItemExpr(field = x)
        │   └── RootExpr
        └── rhs: LiteralExpr(value: 10i32, dtype: i32)
        ");

        let complex_binary = and(
            eq(get_item("name", root()), lit("alice")),
            gt(get_item("age", root()), lit(18)),
        );
        assert_snapshot!(complex_binary.display_tree().to_string(), @r#"
        BinaryExpr(and)
        ├── lhs: BinaryExpr(=)
        │   ├── lhs: GetItemExpr(field = name)
        │   │   └── RootExpr
        │   └── rhs: LiteralExpr(value: "alice", dtype: utf8)
        └── rhs: BinaryExpr(>)
            ├── lhs: GetItemExpr(field = age)
            │   └── RootExpr
            └── rhs: LiteralExpr(value: 18i32, dtype: i32)
        "#);

        let select_expr = select(["name", "age"], root());
        assert_snapshot!(select_expr.display_tree().to_string(), @r#"
        SelectExpr(include): ["name", "age"]
        └── RootExpr
        "#);

        let select_exclude_expr = select_exclude(["internal_id", "metadata"], root());
        assert_snapshot!(select_exclude_expr.display_tree().to_string(), @r#"
        SelectExpr(exclude): ["internal_id", "metadata"]
        └── RootExpr
        "#);

        let cast_expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_snapshot!(cast_expr.display_tree().to_string(), @r"
        CastExpr(target: i64)
        └── GetItemExpr(field = value)
            └── RootExpr
        ");

        let not_expr = not(eq(get_item("active", root()), lit(true)));
        assert_snapshot!(not_expr.display_tree().to_string(), @r"
        NotExpr
        └── BinaryExpr(=)
            ├── lhs: GetItemExpr(field = active)
            │   └── RootExpr
            └── rhs: LiteralExpr(value: true, dtype: bool)
        ");

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
        BetweenExpr
        ├── array: GetItemExpr(field = score)
        │   └── RootExpr
        ├── lower (NonStrict): LiteralExpr(value: 0i32, dtype: i32)
        └── upper (NonStrict): LiteralExpr(value: 100i32, dtype: i32)
        ");

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
        SelectExpr(include): ["result"]
        └── CastExpr(target: bool)
            └── BetweenExpr
                ├── array: GetItemExpr(field = score)
                │   └── RootExpr
                ├── lower (Strict): LiteralExpr(value: 50i32, dtype: i32)
                └── upper (NonStrict): LiteralExpr(value: 100i32, dtype: i32)
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
        SelectExpr(include): ["fizz", "buzz"]
        └── PackExpr
            ├── fizz: RootExpr
            ├── bar: LiteralExpr(value: 5i32, dtype: i32)
            └── buzz: BinaryExpr(=)
                ├── lhs: LiteralExpr(value: 42i32, dtype: i32)
                └── rhs: GetItemExpr(field = answer)
                    └── RootExpr
        "#);
    }
}
