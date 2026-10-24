// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression display, binding and optimizer tests that involve this crate's functions.

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_expr::and;
use vortex_expr::bound;
use vortex_expr::cast;
use vortex_expr::col;
use vortex_expr::eq;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::gt_eq;
use vortex_expr::lit;
use vortex_expr::lt;
use vortex_expr::lt_eq;
use vortex_expr::not;
use vortex_expr::not_eq;
use vortex_expr::or;
use vortex_expr::pack;
use vortex_expr::root;
use vortex_expr::select;
use vortex_expr::select_exclude;

use crate::between::BetweenOptions;
use crate::between::StrictComparison;
use crate::expr::between;
use crate::expr::bound::case_when as bound_case_when;
use crate::expr::case_when;
use crate::expr::merge;

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

/// `Select` over a `pack` rewrites to a `pack` of the selected child fields. It must read
/// those fields out of the child `pack` itself, because the optimizer visits children before
/// their parent and so never revisits the children a rule introduces.
#[test]
fn optimize_select_of_pack_reads_pack_fields() -> VortexResult<()> {
    let field_names = (0..4)
        .map(|idx| FieldName::from(format!("field_{idx}")))
        .collect::<Vec<_>>();
    let fields = StructFields::new(
        field_names.clone().into(),
        vec![DType::Primitive(PType::U64, Nullability::NonNullable); field_names.len()],
    );
    let scope = DType::Struct(fields, Nullability::NonNullable);

    let root_pack = pack(
        field_names
            .iter()
            .map(|name| (name.clone(), get_item(name.clone(), root())))
            .collect::<Vec<_>>(),
        Nullability::NonNullable,
    );
    let expr = select(
        field_names
            .iter()
            .cloned()
            .chain(["input_row_idx".into(), "input_file_idx".into()])
            .collect::<Vec<_>>(),
        merge([
            root_pack,
            pack(
                [
                    ("input_row_idx", lit(0_u64)),
                    ("input_file_idx", lit(1_u64)),
                ],
                Nullability::NonNullable,
            ),
        ]),
    );

    let expected = pack(
        field_names
            .into_iter()
            .map(|name| (name.clone(), get_item(name, root())))
            .chain([
                ("input_row_idx".into(), lit(0_u64)),
                ("input_file_idx".into(), lit(1_u64)),
            ])
            .collect::<Vec<_>>(),
        Nullability::NonNullable,
    );

    let expected = expected.bind(&scope)?;

    let optimized = expr.bind(&scope)?.optimize_recursive()?;
    assert_eq!(optimized, expected, "got {optimized}");
    assert_eq!(optimized.optimize_recursive()?, expected);

    Ok(())
}

/// `Merge` reduces to a `pack` that reads each field out of its children, so it must also
/// read straight out of a `pack` child.
#[test]
fn optimize_merge_of_packs_reads_pack_fields() -> VortexResult<()> {
    let scope = DType::Struct(StructFields::empty(), Nullability::NonNullable);
    let expr = merge([
        pack([("a", lit(0_u64))], Nullability::NonNullable),
        pack([("b", lit(1_u64))], Nullability::NonNullable),
    ]);

    let expected = pack(
        [("a", lit(0_u64)), ("b", lit(1_u64))],
        Nullability::NonNullable,
    );

    let expected = expected.bind(&scope)?;

    let optimized = expr.bind(&scope)?.optimize_recursive()?;
    assert_eq!(optimized, expected, "got {optimized}");
    assert_eq!(optimized.optimize_recursive()?, expected);

    Ok(())
}

#[test]
fn bound_constructors_preserve_order_and_types() -> VortexResult<()> {
    let value_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let scope = DType::Struct(
        StructFields::from_iter([("value", value_dtype.clone())]),
        Nullability::NonNullable,
    );

    let root = bound::root(scope.clone());
    let value = bound::get_item("value", root);
    let literal = bound::lit(5i32);
    let condition = bound::gt(value.clone(), literal.clone());
    assert_eq!(condition.dtype(), &DType::Bool(Nullability::NonNullable));
    assert_eq!(condition.children(), &[value.clone(), literal.clone()]);

    let case = bound_case_when(condition.clone(), value.clone(), literal.clone());
    assert_eq!(case.dtype(), &value_dtype);
    assert_eq!(case.children(), &[condition.clone(), value, literal]);

    let packed = bound::pack(
        [("condition", condition.clone()), ("value", case.clone())],
        Nullability::NonNullable,
    );
    assert_eq!(packed.children(), &[condition, case.clone()]);
    assert_eq!(
        packed.dtype(),
        &DType::Struct(
            StructFields::from_iter([
                ("condition", DType::Bool(Nullability::NonNullable)),
                ("value", value_dtype),
            ]),
            Nullability::NonNullable,
        )
    );

    let unbound = case_when(gt(col("value"), lit(5i32)), col("value"), lit(5i32));
    assert_eq!(unbound.bind(&scope)?, case);
    Ok(())
}

#[test]
fn expr_display() {
    assert_eq!(col("a").to_string(), "$.a");
    assert_eq!(root().to_string(), "$");

    let col1: Expression = col("col1");
    let col2: Expression = col("col2");
    assert_eq!(
        and(col1.clone(), col2.clone()).to_string(),
        "($.col1 and $.col2)"
    );
    assert_eq!(
        or(col1.clone(), col2.clone()).to_string(),
        "($.col1 or $.col2)"
    );
    assert_eq!(
        eq(col1.clone(), col2.clone()).to_string(),
        "($.col1 = $.col2)"
    );
    assert_eq!(
        not_eq(col1.clone(), col2.clone()).to_string(),
        "($.col1 != $.col2)"
    );
    assert_eq!(
        gt(col1.clone(), col2.clone()).to_string(),
        "($.col1 > $.col2)"
    );
    assert_eq!(
        gt_eq(col1.clone(), col2.clone()).to_string(),
        "($.col1 >= $.col2)"
    );
    assert_eq!(
        lt(col1.clone(), col2.clone()).to_string(),
        "($.col1 < $.col2)"
    );
    assert_eq!(
        lt_eq(col1.clone(), col2.clone()).to_string(),
        "($.col1 <= $.col2)"
    );

    assert_eq!(
        or(lt(col1.clone(), col2.clone()), not_eq(col1.clone(), col2),).to_string(),
        "(($.col1 < $.col2) or ($.col1 != $.col2))"
    );

    assert_eq!(not(col1).to_string(), "vortex.not($.col1)");

    assert_eq!(
        select(vec![FieldName::from("col1")], root()).to_string(),
        "${col1}"
    );
    assert_eq!(
        select(
            vec![FieldName::from("col1"), FieldName::from("col2")],
            root()
        )
        .to_string(),
        "${col1, col2}"
    );
    assert_eq!(
        select_exclude(
            vec![FieldName::from("col1"), FieldName::from("col2")],
            root()
        )
        .to_string(),
        "${~ col1, col2}"
    );

    assert_eq!(lit(Scalar::from(0u8)).to_string(), "0u8");
    assert_eq!(lit(Scalar::from(0.0f32)).to_string(), "0f32");
    assert_eq!(
        lit(Scalar::from(i64::MAX)).to_string(),
        "9223372036854775807i64"
    );
    assert_eq!(lit(Scalar::from(true)).to_string(), "true");
    assert_eq!(
        lit(Scalar::null(DType::Bool(Nullability::Nullable))).to_string(),
        "null"
    );

    assert_eq!(
        lit(Scalar::struct_(
            DType::Struct(
                StructFields::new(
                    FieldNames::from(["dog", "cat"]),
                    vec![
                        DType::Primitive(PType::U32, Nullability::NonNullable),
                        DType::Utf8(Nullability::NonNullable)
                    ],
                ),
                Nullability::NonNullable
            ),
            vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
        ))
        .to_string(),
        "{dog: 32u32, cat: \"rufus\"}"
    );
}
