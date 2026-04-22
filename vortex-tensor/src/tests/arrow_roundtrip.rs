// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow ↔ DType round-trip tests for tensor extension types.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::arrow::FromArrowWithSession;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;

use crate::tests::SESSION;
use crate::types::fixed_shape::FixedShapeTensor;
use crate::types::fixed_shape::FixedShapeTensorMetadata;
use crate::types::vector::Vector;

const VECTOR_EXT_NAME: &str = "vortex.tensor.vector";
const FIXED_SHAPE_EXT_NAME: &str = "vortex.fixed_shape_tensor";

fn vector_dtype(len: u32) -> DType {
    let storage = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
        len,
        Nullability::NonNullable,
    );
    let ext = ExtDType::<Vector>::try_new(vortex_array::extension::EmptyMetadata, storage).unwrap();
    DType::Extension(ext.erased())
}

fn fixed_shape_dtype(metadata: FixedShapeTensorMetadata, element_count: u32) -> DType {
    let storage = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
        element_count,
        Nullability::NonNullable,
    );
    let ext = ExtDType::<FixedShapeTensor>::try_new(metadata, storage).unwrap();
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
        Some(VECTOR_EXT_NAME),
    );
    // EmptyMetadata: no metadata key emitted.
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
fn fixed_shape_tensor_metadata_roundtrip() {
    let metadata = FixedShapeTensorMetadata::new(vec![2, 3, 4])
        .with_dim_names(vec!["x".into(), "y".into(), "z".into()])
        .unwrap()
        .with_permutation(vec![2, 0, 1])
        .unwrap();

    let original = DType::struct_(
        [("tensor", fixed_shape_dtype(metadata, 24))],
        Nullability::NonNullable,
    );

    let schema = original.to_arrow_schema().unwrap();
    let field = schema.field(0);

    assert_eq!(
        field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str),
        Some(FIXED_SHAPE_EXT_NAME),
    );
    assert!(field.metadata().get(EXTENSION_TYPE_METADATA_KEY).is_some());

    let recovered = DType::from_arrow_with_session(&schema, &SESSION);
    assert_eq!(recovered, original);
}

#[test]
fn tensor_inside_nested_struct_roundtrips() {
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
