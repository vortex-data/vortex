// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow ↔ DType round-trip tests for tensor extension types.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::arrow::FromArrowType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::EmptyMetadata;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::validity::Validity;

use crate::tests::SESSION;
use crate::types::vector::Vector;

fn vector_dtype(len: u32) -> DType {
    let storage = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
        len,
        Nullability::NonNullable,
    );
    let ext = ExtDType::<Vector>::try_new(EmptyMetadata, storage).unwrap();
    DType::Extension(ext.erased())
}

#[test]
fn vector_forward_carries_extension_name() {
    let original = DType::struct_([("embedding", vector_dtype(4))], Nullability::NonNullable);

    let schema = original.to_arrow_schema().unwrap();
    let field = schema.field(0);

    assert_eq!(
        field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str),
        Some(Vector.id().as_str()),
    );
    // EmptyMetadata → no metadata key.
    assert!(field.metadata().get(EXTENSION_TYPE_METADATA_KEY).is_none());

    let DataType::FixedSizeList(element, size) = field.data_type() else {
        panic!("expected FixedSizeList, got {:?}", field.data_type());
    };
    assert_eq!(*size, 4);
    assert_eq!(element.data_type(), &DataType::Float32);
}

#[test]
fn vector_roundtrip_with_session() {
    let original = DType::struct_([("embedding", vector_dtype(128))], Nullability::NonNullable);

    let schema = original.to_arrow_schema().unwrap();
    let recovered = DType::from_arrow_with_session(&schema, &SESSION);

    assert_eq!(recovered, original);
}

#[test]
fn vector_without_registration_falls_back_to_fsl() {
    let original = DType::struct_([("embedding", vector_dtype(16))], Nullability::NonNullable);

    let empty_session = vortex_session::VortexSession::empty();
    let schema = original.to_arrow_schema().unwrap();
    let recovered = DType::from_arrow_with_session(&schema, &empty_session);

    let expected = DType::struct_(
        [(
            "embedding",
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                16,
                Nullability::NonNullable,
            ),
        )],
        Nullability::NonNullable,
    );
    assert_eq!(recovered, expected);
}

#[test]
fn vector_inside_nested_struct_roundtrips() {
    let inner = DType::struct_([("embedding", vector_dtype(8))], Nullability::Nullable);
    let original = DType::struct_(
        [("inner", inner), ("id", DType::Utf8(Nullability::Nullable))],
        Nullability::NonNullable,
    );

    let schema = original.to_arrow_schema().unwrap();
    let recovered = DType::from_arrow_with_session(&schema, &SESSION);

    assert_eq!(recovered, original);
}

#[test]
fn vector_record_batch_round_trip_carries_field_metadata() {
    let vector_array = Vector::constant_array(&[1.0f32, 2.0, 3.0, 4.0], 2).unwrap();
    let struct_array = StructArray::from_fields(&[("embedding", vector_array)]).unwrap();

    let dtype = DType::struct_([("embedding", vector_dtype(4))], Nullability::NonNullable);
    let schema = dtype.to_arrow_schema().unwrap();
    let rb = struct_array
        .into_record_batch_with_schema_with_session(&schema, &SESSION)
        .unwrap();

    let column = rb.column(0);
    let DataType::FixedSizeList(_, size) = column.data_type() else {
        panic!(
            "expected storage FixedSizeList, got {:?}",
            column.data_type()
        );
    };
    assert_eq!(*size, 4);

    assert_eq!(
        rb.schema()
            .field(0)
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str),
        Some(Vector.id().as_str()),
    );
}

#[test]
fn temporal_extension_still_uses_native_arrow() {
    let ts = Timestamp::new_with_tz(TimeUnit::Microseconds, None, Nullability::Nullable);
    let original = DType::struct_(
        [("ts", DType::Extension(ts.erased()))],
        Nullability::NonNullable,
    );

    let schema = original.to_arrow_schema().unwrap();
    let field = schema.field(0);

    assert!(matches!(
        field.data_type(),
        DataType::Timestamp(ArrowTimeUnit::Microsecond, None)
    ));
    assert!(field.metadata().get(EXTENSION_TYPE_NAME_KEY).is_none());
    assert!(field.metadata().get(EXTENSION_TYPE_METADATA_KEY).is_none());
}

/// Build a storage FSL<f32> with `num_rows` rows, each of `elements_per_row` elements.
fn fsl_f32_storage(elements_per_row: u32, num_rows: usize) -> ArrayRef {
    let total = elements_per_row as usize * num_rows;
    let elements = PrimitiveArray::from_iter((0..total).map(|i| i as f32));
    FixedSizeListArray::try_new(
        elements.into_array(),
        elements_per_row,
        Validity::NonNullable,
        num_rows,
    )
    .unwrap()
    .into_array()
}

#[test]
fn vector_record_batch_round_trip() {
    let vector_array =
        ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, fsl_f32_storage(4, 2))
            .unwrap()
            .into_array();
    let original = StructArray::from_fields(&[("embedding", vector_array)]).unwrap();

    let dtype = DType::struct_([("embedding", vector_dtype(4))], Nullability::NonNullable);
    let schema = dtype.to_arrow_schema().unwrap();
    let rb = original
        .into_record_batch_with_schema_with_session(&schema, &SESSION)
        .unwrap();

    let recovered = ArrayRef::from_arrow_with_session(rb, false, &SESSION).unwrap();
    assert_eq!(recovered.dtype(), &dtype);
}
