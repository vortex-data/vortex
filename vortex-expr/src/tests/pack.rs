// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::builtin::pack::Pack;
use vortex_array::scalar_fn::builtin::pack::PackOptions;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::ScalarFnExprExt;
use crate::col;
use crate::pack;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn test_array() -> ArrayRef {
    StructArray::from_fields(&[
        ("a", buffer![0, 1, 2].into_array()),
        ("b", buffer![4, 5, 6].into_array()),
    ])
    .unwrap()
    .into_array()
}

fn field_array(array: &ArrayRef, field_path: &[&str]) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut field_path = field_path.iter();

    let Some(field) = field_path.next() else {
        vortex_bail!("empty field path");
    };

    let mut array = array
        .clone()
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name(field)?
        .clone();
    for field in field_path {
        let next = array
            .clone()
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name(field)?
            .clone();
        array = next;
    }
    Ok(array)
}

#[test]
pub fn test_empty_pack() {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = Pack.new_expr(
        PackOptions {
            names: Default::default(),
            nullability: Default::default(),
        },
        [],
    );

    let test_array = test_array();
    let actual_array = test_array.clone().apply(&expr).unwrap();
    assert_eq!(actual_array.len(), test_array.len());
    let nfields = actual_array
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .struct_fields()
        .nfields();
    assert_eq!(nfields, 0);
}

#[test]
pub fn test_simple_pack() {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = Pack.new_expr(
        PackOptions {
            names: ["one", "two", "three"].into(),
            nullability: Nullability::NonNullable,
        },
        [col("a"), col("b"), col("a")],
    );

    let actual_array = test_array()
        .apply(&expr)
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_array.names(), ["one", "two", "three"]);
    assert!(matches!(actual_array.validity(), Ok(Validity::NonNullable)));

    assert_arrays_eq!(
        field_array(&actual_array.clone().into_array(), &["one"]).unwrap(),
        PrimitiveArray::from_iter([0i32, 1, 2]),
        &mut ctx
    );
    assert_arrays_eq!(
        field_array(&actual_array.clone().into_array(), &["two"]).unwrap(),
        PrimitiveArray::from_iter([4i32, 5, 6]),
        &mut ctx
    );
    assert_arrays_eq!(
        field_array(&actual_array.into_array(), &["three"]).unwrap(),
        PrimitiveArray::from_iter([0i32, 1, 2]),
        &mut ctx
    );
}

#[test]
pub fn test_nested_pack() {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = Pack.new_expr(
        PackOptions {
            names: ["one", "two", "three"].into(),
            nullability: Nullability::NonNullable,
        },
        [
            col("a"),
            Pack.new_expr(
                PackOptions {
                    names: ["two_one", "two_two"].into(),
                    nullability: Nullability::NonNullable,
                },
                [col("b"), col("b")],
            ),
            col("a"),
        ],
    );

    let actual_array = test_array()
        .apply(&expr)
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_array.names(), ["one", "two", "three"]);

    assert_arrays_eq!(
        field_array(&actual_array.clone().into_array(), &["one"]).unwrap(),
        PrimitiveArray::from_iter([0i32, 1, 2]),
        &mut ctx
    );
    assert_arrays_eq!(
        field_array(&actual_array.clone().into_array(), &["two", "two_one"]).unwrap(),
        PrimitiveArray::from_iter([4i32, 5, 6]),
        &mut ctx
    );
    assert_arrays_eq!(
        field_array(&actual_array.clone().into_array(), &["two", "two_two"]).unwrap(),
        PrimitiveArray::from_iter([4i32, 5, 6]),
        &mut ctx
    );
    assert_arrays_eq!(
        field_array(&actual_array.into_array(), &["three"]).unwrap(),
        PrimitiveArray::from_iter([0i32, 1, 2]),
        &mut ctx
    );
}

#[test]
pub fn test_pack_nullable() {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = Pack.new_expr(
        PackOptions {
            names: ["one", "two", "three"].into(),
            nullability: Nullability::Nullable,
        },
        [col("a"), col("b"), col("a")],
    );

    let actual_array = test_array()
        .apply(&expr)
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_array.names(), ["one", "two", "three"]);
    assert!(matches!(actual_array.validity(), Ok(Validity::AllValid)));
    let expected = StructArray::try_new(
        ["one", "two", "three"].into(),
        vec![
            buffer![0i32, 1, 2].into_array(),
            buffer![4i32, 5, 6].into_array(),
            buffer![0i32, 1, 2].into_array(),
        ],
        3,
        Validity::AllValid,
    )
    .unwrap();
    assert_arrays_eq!(actual_array, expected, &mut ctx);
}

#[test]
pub fn test_display() {
    let expr = pack(
        [("id", col("user_id")), ("name", col("username"))],
        Nullability::NonNullable,
    );
    assert_eq!(expr.to_string(), "pack(id: $.user_id, name: $.username)");

    let expr2 = Pack.new_expr(
        PackOptions {
            names: ["x", "y"].into(),
            nullability: Nullability::Nullable,
        },
        [col("a"), col("b")],
    );
    assert_eq!(expr2.to_string(), "pack(x: $.a, y: $.b)?");
}
