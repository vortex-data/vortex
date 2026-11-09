// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType::I32;
use vortex_array::dtype::StructFields;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::get_item;
use vortex_expr::proto::ExprSerializeProtoExt;
use vortex_expr::root;
use vortex_expr::test_harness;
use vortex_session::VortexSession;

use crate::expr::select;
use crate::expr::select_exclude;
use crate::select::FieldSelection;
use crate::select::Select;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();

    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

fn test_array() -> StructArray {
    StructArray::from_fields(&[
        ("a", buffer![0, 1, 2].into_array()),
        ("b", buffer![4, 5, 6].into_array()),
    ])
    .unwrap()
}

#[test]
pub fn include_columns() {
    let mut ctx = SESSION.create_execution_ctx();
    let st = test_array();
    let select = select(vec![FieldName::from("a")], root());
    let selected = st.into_array().apply(&select).unwrap();
    let selected_names = selected.dtype().as_struct_fields().names();
    assert_eq!(selected_names.as_ref(), &["a"]);
    assert_arrays_eq!(
        selected,
        StructArray::from_fields(&[("a", buffer![0i32, 1, 2].into_array())]).unwrap(),
        &mut ctx
    );
}

#[test]
pub fn exclude_columns() {
    let mut ctx = SESSION.create_execution_ctx();
    let st = test_array();
    let select = select_exclude(vec![FieldName::from("a")], root());
    let selected = st.into_array().apply(&select).unwrap();
    let selected_names = selected.dtype().as_struct_fields().names();
    assert_eq!(selected_names.as_ref(), &["b"]);
    assert_arrays_eq!(
        selected,
        StructArray::from_fields(&[("b", buffer![4i32, 5, 6].into_array())]).unwrap(),
        &mut ctx
    );
}

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();

    let select_expr = select(vec![FieldName::from("a")], root());
    let expected_dtype = DType::Struct(
        dtype
            .as_struct_fields_opt()
            .unwrap()
            .project(&["a".into()])
            .unwrap(),
        Nullability::NonNullable,
    );
    assert_eq!(select_expr.return_dtype(&dtype).unwrap(), expected_dtype);

    let select_expr_exclude = select_exclude(
        vec![
            FieldName::from("col1"),
            FieldName::from("col2"),
            FieldName::from("bool1"),
            FieldName::from("bool2"),
        ],
        root(),
    );
    assert_eq!(
        select_expr_exclude.return_dtype(&dtype).unwrap(),
        expected_dtype
    );

    let select_expr_exclude = select_exclude(
        vec![FieldName::from("col1"), FieldName::from("col2")],
        root(),
    );
    assert_eq!(
        select_expr_exclude.return_dtype(&dtype).unwrap(),
        DType::Struct(
            dtype
                .as_struct_fields_opt()
                .unwrap()
                .project(&["a".into(), "bool1".into(), "bool2".into()])
                .unwrap(),
            Nullability::NonNullable
        )
    );
}

#[test]
fn test_as_include_names() {
    let field_names = FieldNames::from(["a", "b", "c"]);
    for expr in [select(["a"], root()), select_exclude(["b", "c"], root())] {
        assert_eq!(
            expr.as_::<Select>()
                .normalize_to_included_fields(&field_names)
                .unwrap(),
            FieldNames::from(["a"])
        );
    }
}

#[test]
fn execute_constant_struct_stays_constant() -> VortexResult<()> {
    let field_count = 128usize;
    let names = (0..field_count)
        .map(|idx| FieldName::from(format!("f{idx}")))
        .collect::<Vec<_>>();
    let dtypes = vec![I32.into(); field_count];
    let fields = StructFields::new(FieldNames::from(names), dtypes);
    let scalar = Scalar::struct_(
        DType::Struct(fields, Nullability::NonNullable),
        (0..128).map(Scalar::from),
    );
    let array = ConstantArray::new(scalar, 1_000_000).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let item: ArrayRef = ScalarFnArray::try_new(
        Select.bind(FieldSelection::Include(FieldNames::from(["f97", "f3"]))),
        vec![array],
    )?
    .into_array()
    .execute(&mut ctx)?;

    let constant = item.as_::<Constant>();
    assert_eq!(constant.len(), 1_000_000);
    assert_eq!(
        constant.scalar(),
        &Scalar::struct_(
            DType::Struct(
                StructFields::new(
                    FieldNames::from(["f97", "f3"]),
                    vec![I32.into(), I32.into()],
                ),
                Nullability::NonNullable,
            ),
            [Scalar::from(97), Scalar::from(3)],
        )
    );
    Ok(())
}

#[test]
fn test_remove_select_rule() {
    let dtype = DType::Struct(
        StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
        Nullable,
    );
    let e = select(["a", "b"], root());

    let result = e.bind(&dtype).unwrap().optimize_recursive().unwrap();

    assert_eq!(result.dtype(), &dtype);
}

#[test]
fn test_remove_select_rule_exclude_fields() {
    use crate::expr::select_exclude;

    let dtype = DType::Struct(
        StructFields::new(
            ["a", "b", "c"].into(),
            vec![I32.into(), I32.into(), I32.into()],
        ),
        Nullable,
    );
    let e = select_exclude(["c"], root());

    let result = e.bind(&dtype).unwrap().optimize_recursive().unwrap();

    // Should exclude "c" and include "a" and "b"
    let result_dtype = result.dtype();
    assert!(result_dtype.is_nullable());
    let fields = result_dtype.as_struct_fields_opt().unwrap();
    assert_eq!(fields.names().as_ref(), &["a", "b"]);
    assert_eq!(
        result_dtype,
        &DType::struct_([("a", I32), ("b", I32)], Nullable)
    );
}

#[test]
fn test_roundtrip_preserves_field_names() -> VortexResult<()> {
    let name = "quote\"slash\\line\n\t\r\0 ♡";
    let escaped_name = name.escape_debug().to_string();
    let input = StructArray::from_fields(&[
        (name, buffer![1i32, 2].into_array()),
        (escaped_name.as_str(), buffer![3i32, 4].into_array()),
    ])?
    .into_array();
    for (expression, expected_name, expected_values) in [
        (select([name], root()), name, buffer![1i32, 2].into_array()),
        (
            select_exclude([name], root()),
            escaped_name.as_str(),
            buffer![3i32, 4].into_array(),
        ),
    ] {
        let restored = Expression::from_proto(&expression.serialize_proto()?, &SESSION)?;
        let expected = StructArray::from_fields(&[(expected_name, expected_values)])?;
        assert_arrays_eq!(
            input.clone().apply(&restored)?,
            expected,
            &mut SESSION.create_execution_ctx()
        );
    }
    Ok(())
}

#[rstest]
#[case::include(FieldSelection::Include(FieldNames::from(["a"])))]
#[case::exclude(FieldSelection::Exclude(FieldNames::from(["b"])))]
fn get_item_reduces_through_select(
    #[case] selection: FieldSelection,
    #[values(
        Validity::NonNullable,
        Validity::AllValid,
        Validity::AllInvalid,
        Validity::from_iter([true, false, true])
    )]
    validity: Validity,
) -> VortexResult<()> {
    let array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            ConstantArray::new(42i32, 3).into_array(),
            buffer![1i32, 2, 3].into_array(),
        ],
        3,
        validity,
    )?
    .into_array();
    let expected = array.get_item("a")?.optimize_ctx(&SESSION)?;
    let selected = ScalarFnArray::try_new(Select.bind(selection), [array])?.into_array();
    let item = GetItem::try_new(selected, "a")?
        .into_array()
        .optimize_ctx(&SESSION)?;

    assert_eq!(item.encoding_id(), expected.encoding_id());
    assert_eq!(item.dtype(), expected.dtype());
    assert_arrays_eq!(item, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::include(select(["a"], root()))]
#[case::exclude(select_exclude(["b"], root()))]
fn get_item_select_expression_reduces(#[case] selected: Expression) -> VortexResult<()> {
    let dtype = test_array().dtype().clone();
    let result = get_item("a", selected).bind(&dtype)?.optimize_recursive()?;
    assert_eq!(result, get_item("a", root()).bind(&dtype)?);
    Ok(())
}

#[rstest]
#[case::include(select(["b"], root()))]
#[case::exclude(select_exclude(["a"], root()))]
fn get_item_select_rejects_removed_field(#[case] selected: Expression) {
    assert!(
        test_array()
            .into_array()
            .apply(&get_item("a", selected))
            .is_err()
    );
}
