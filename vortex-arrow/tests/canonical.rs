// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

//! Round-trip tests between canonical Vortex arrays and Arrow, moved from
//! `vortex-array/src/canonical.rs` when Arrow interoperability moved into this crate.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ListArray as ArrowListArray;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt64Type;
use arrow_buffer::NullBufferBuilder;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::FromArrowArray;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

/// A shared session for these canonical tests, used to create execution contexts.
static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_canonicalize_nested_struct() {
    let mut ctx = SESSION.create_execution_ctx();
    // Create a struct array with multiple internal components.
    let nested_struct_array = StructArray::from_fields(&[
        ("a", buffer![1u64].into_array()),
        (
            "b",
            StructArray::from_fields(&[(
                "inner_a",
                // The nested struct contains a ConstantArray representing the primitive array
                //   [100i64]
                // ConstantArray is not a canonical type, so converting `into_arrow()` should
                // map this to the nearest canonical type (PrimitiveArray).
                ConstantArray::new(100i64, 1).into_array(),
            )])
            .unwrap()
            .into_array(),
        ),
    ])
    .unwrap();

    let arrow_struct = SESSION
        .arrow()
        .execute_arrow(nested_struct_array.into_array(), None, &mut ctx)
        .unwrap()
        .as_any()
        .downcast_ref::<ArrowStructArray>()
        .cloned()
        .unwrap();

    assert!(
        arrow_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<UInt64Type>>()
            .is_some()
    );

    let inner_struct = Arc::clone(arrow_struct.column(1))
        .as_any()
        .downcast_ref::<ArrowStructArray>()
        .cloned()
        .unwrap();

    let inner_a = inner_struct
        .column(0)
        .as_any()
        .downcast_ref::<ArrowPrimitiveArray<Int64Type>>();
    assert!(inner_a.is_some());

    assert_eq!(
        inner_a.cloned().unwrap(),
        ArrowPrimitiveArray::from_iter([100i64])
    );
}

#[test]
fn roundtrip_struct() {
    let mut ctx = SESSION.create_execution_ctx();
    let mut nulls = NullBufferBuilder::new(6);
    nulls.append_n_non_nulls(4);
    nulls.append_null();
    nulls.append_non_null();
    let names = Arc::new(StringViewArray::from_iter(vec![
        Some("Joseph"),
        None,
        Some("Angela"),
        Some("Mikhail"),
        None,
        None,
    ]));
    let ages = Arc::new(ArrowPrimitiveArray::<Int32Type>::from(vec![
        Some(25),
        Some(31),
        None,
        Some(57),
        None,
        None,
    ]));

    let arrow_struct = ArrowStructArray::new(
        vec![
            Arc::new(Field::new("name", DataType::Utf8View, true)),
            Arc::new(Field::new("age", DataType::Int32, true)),
        ]
        .into(),
        vec![names, ages],
        nulls.finish(),
    );

    let vortex_struct = ArrayRef::from_arrow(&arrow_struct, true).unwrap();
    let vortex_struct = SESSION
        .arrow()
        .execute_arrow(vortex_struct, None, &mut ctx)
        .unwrap();
    assert_eq!(&arrow_struct, vortex_struct.as_struct());
}

#[test]
fn roundtrip_list() {
    let mut ctx = SESSION.create_execution_ctx();
    let names = Arc::new(StringArray::from_iter(vec![
        Some("Joseph"),
        Some("Angela"),
        Some("Mikhail"),
    ]));

    let arrow_list = ArrowListArray::new(
        Arc::new(Field::new_list_field(DataType::Utf8, true)),
        OffsetBuffer::from_lengths(vec![0, 2, 1]),
        names,
        None,
    );
    let list_data_type = arrow_list.data_type();
    let list_field = Field::new(String::new(), list_data_type.clone(), true);

    let vortex_list = ArrayRef::from_arrow(&arrow_list, true).unwrap();

    let rt_arrow_list = SESSION
        .arrow()
        .execute_arrow(vortex_list, Some(&list_field), &mut ctx)
        .unwrap();

    assert_eq!(
        (Arc::new(arrow_list.clone()) as ArrowArrayRef).as_ref(),
        rt_arrow_list.as_ref()
    );
}
