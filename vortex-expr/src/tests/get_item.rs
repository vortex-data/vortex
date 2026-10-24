// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::Slice;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::checked_add;
use crate::get_item;
use crate::lit;
use crate::pack;
use crate::root;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn test_array() -> StructArray {
    StructArray::from_fields(&[
        ("a", buffer![0i32, 1, 2].into_array()),
        ("b", buffer![4i64, 5, 6].into_array()),
    ])
    .unwrap()
}

#[test]
fn get_item_by_name() {
    let st = test_array();
    let get_item = get_item("a", root());
    assert!(
        get_item
            .as_scalar()
            .is_some_and(|f| f.signature().is_strict())
    );
    let item = st.into_array().apply(&get_item).unwrap();
    assert_eq!(item.dtype(), &DType::from(PType::I32));
    assert_arrays_eq!(
        item,
        buffer![0i32, 1, 2].into_array(),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn get_item_by_name_none() {
    let st = test_array();
    let get_item = get_item("c", root());
    assert!(st.into_array().apply(&get_item).is_err());
}

#[test]
fn get_nullable_field() {
    let st = StructArray::try_new(
        FieldNames::from(["a"]),
        vec![buffer![1i32].into_array()],
        1,
        Validity::AllInvalid,
    )
    .unwrap()
    .into_array();

    let get_item_expr = get_item("a", root());
    let item = st.apply(&get_item_expr).unwrap();
    // The dtype should be nullable since it inherits struct validity
    assert_eq!(
        item.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert_arrays_eq!(
        item,
        PrimitiveArray::from_option_iter([None::<i32>]),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn get_non_nullable_field_from_all_valid_nullable_struct() -> VortexResult<()> {
    let st = StructArray::try_new(
        FieldNames::from(["a"]),
        vec![buffer![1i32].into_array()],
        1,
        Validity::AllValid,
    )?
    .into_array();

    let item = st.apply(&get_item("a", root()))?;
    assert_eq!(
        item.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert_arrays_eq!(
        item,
        PrimitiveArray::from_option_iter([Some(1i32)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn execute_constant_struct_stays_constant() -> VortexResult<()> {
    let field_count = 128usize;
    let selected_idx = 97i32;
    let names = (0..field_count)
        .map(|idx| FieldName::from(format!("f{idx}")))
        .collect::<Vec<_>>();
    let dtypes = vec![DType::Primitive(PType::I32, NonNullable); field_count];
    let fields = StructFields::new(FieldNames::from(names), dtypes);
    let scalar = Scalar::struct_(
        DType::Struct(fields, NonNullable),
        (0..128).map(|idx| Scalar::primitive(idx, NonNullable)),
    );
    let array = ConstantArray::new(scalar, 1_000_000).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let item: ArrayRef = GetItem::try_new(array, "f97")?
        .into_array()
        .execute(&mut ctx)?;

    let constant = item.as_::<Constant>();
    assert_eq!(constant.len(), 1_000_000);
    assert_eq!(
        constant.scalar(),
        &Scalar::primitive(selected_idx, NonNullable)
    );
    Ok(())
}

#[test]
fn get_item_pushes_through_filter() -> VortexResult<()> {
    use vortex_mask::Mask;

    let filtered = test_array()
        .into_array()
        .filter(Mask::from_iter([true, false, true]))?;

    let item = filtered.get_item("a")?;

    assert!(item.is::<Filter>());
    assert!(!item.is::<ScalarFn>());
    assert_arrays_eq!(
        item,
        buffer![0i32, 2].into_array(),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn get_item_pushes_through_slice() -> VortexResult<()> {
    let sliced = test_array().into_array().slice(1..3)?;

    let item = sliced.get_item("a")?;

    assert!(item.is::<Primitive>());
    assert!(!item.is::<Slice>());
    assert!(!item.is::<ScalarFn>());
    assert_arrays_eq!(
        item,
        buffer![1i32, 2].into_array(),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn get_item_pushes_through_masked() -> VortexResult<()> {
    let masked = test_array()
        .into_array()
        .mask(Validity::from_iter([true, false, true]).to_array(3))?;

    let item = masked.get_item("a")?;

    assert!(item.is::<Primitive>());
    assert_eq!(
        item.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert!(!item.is::<ScalarFn>());
    assert_arrays_eq!(
        item,
        PrimitiveArray::from_option_iter([Some(0i32), None, Some(2)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn test_pack_get_item_rule() {
    // Create: pack(a: lit(1), b: lit(2)).get_item("b")
    let pack_expr = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
    let get_item_expr = get_item("b", pack_expr);

    let dtype = DType::Struct(StructFields::empty(), NonNullable);
    let result = get_item_expr
        .bind(&dtype)
        .and_then(|expr| expr.optimize_recursive())
        .unwrap();

    assert_eq!(result, lit(2).bind(&dtype).unwrap());
}

#[test]
fn test_multi_level_pack_get_item_simplify() {
    let inner_pack = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
    let get_a = get_item("a", inner_pack);

    let outer_pack = pack([("x", get_a), ("y", lit(3)), ("z", lit(4))], NonNullable);
    let get_z = get_item("z", outer_pack);

    let dtype = DType::Primitive(PType::I32, NonNullable);

    let result = get_z.bind(&dtype).unwrap().optimize_recursive().unwrap();
    assert_eq!(result, lit(4).bind(&dtype).unwrap());
}

#[test]
fn test_deeply_nested_pack_get_item() {
    let innermost = pack([("a", lit(42))], NonNullable);
    let get_a = get_item("a", innermost);

    let level2 = pack([("b", get_a)], NonNullable);
    let get_b = get_item("b", level2);

    let level3 = pack([("c", get_b)], NonNullable);
    let get_c = get_item("c", level3);

    let outermost = pack([("final", get_c)], NonNullable);
    let get_final = get_item("final", outermost);

    let dtype = DType::Primitive(PType::I32, NonNullable);

    let result = get_final
        .bind(&dtype)
        .unwrap()
        .optimize_recursive()
        .unwrap();
    assert_eq!(result, lit(42).bind(&dtype).unwrap());
}

#[test]
fn test_partial_pack_get_item_simplify() {
    let inner_pack = pack([("x", lit(1)), ("y", lit(2))], NonNullable);
    let get_x = get_item("x", inner_pack);
    let add_expr = checked_add(get_x, lit(10));

    let outer_pack = pack([("result", add_expr)], NonNullable);
    let get_result = get_item("result", outer_pack);

    let dtype = DType::Primitive(PType::I32, NonNullable);

    let result = get_result
        .bind(&dtype)
        .unwrap()
        .optimize_recursive()
        .unwrap();
    let expected = checked_add(lit(1), lit(10)).bind(&dtype).unwrap();
    assert_eq!(&result, &expected);
}

#[test]
fn get_item_filter_list_field() {
    let mut ctx = SESSION.create_execution_ctx();
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FilterArray;
    use vortex_array::arrays::ListArray;
    use vortex_mask::Mask;

    let list = ListArray::try_new(
        buffer![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.].into_array(),
        buffer![2u64, 4, 6, 8, 10, 12].into_array(),
        Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array()),
        &mut ctx,
    )
    .unwrap();

    let filtered = FilterArray::try_new(
        list.into_array(),
        Mask::from_iter([true, true, false, false, false]),
    )
    .unwrap();

    let st = StructArray::try_new(
        FieldNames::from(["data"]),
        vec![filtered.into_array()],
        2,
        Validity::AllValid,
    )
    .unwrap();

    let result = st.into_array().apply(&get_item("data", root())).unwrap();
    let expected = ListArray::try_new(
        buffer![2f32, 3., 4., 5.].into_array(),
        buffer![0u64, 2, 4].into_array(),
        Validity::AllValid,
        &mut ctx,
    )
    .unwrap();
    assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
}
