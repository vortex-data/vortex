// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::ArrayDeserialization;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionDataParts;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnionVariants;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

mod take;

fn variants() -> VortexResult<UnionVariants> {
    UnionVariants::try_new(
        ["number", "flag"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
        vec![5, 9],
    )
}

fn union_array() -> VortexResult<UnionArray> {
    UnionArray::try_new(
        PrimitiveArray::from_iter([5u8, 9, 5]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32, 0, 30]).into_array(),
            BoolArray::from_iter([false, true, false]).into_array(),
        ],
    )
}

fn nullable_variants() -> VortexResult<UnionVariants> {
    UnionVariants::try_new(
        ["number", "optional"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::Nullable),
        ],
        vec![5, 9],
    )
}

fn nullable_union_array() -> VortexResult<UnionArray> {
    UnionArray::try_new(
        PrimitiveArray::from_option_iter([Some(5u8), None, Some(9), Some(9)]).into_array(),
        nullable_variants()?,
        vec![
            PrimitiveArray::from_iter([10i32, 0, 0, 0]).into_array(),
            PrimitiveArray::from_option_iter([Some(0i64), Some(0), None, Some(40)]).into_array(),
        ],
    )
}

#[test]
fn scalar_at_uses_type_id_indirection() -> VortexResult<()> {
    let array = union_array()?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_eq!(
        array.child_by_name("flag")?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    assert!(array.child_by_name_opt("missing").is_none());
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?
    );
    assert_eq!(
        array.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?
    );
    assert!(matches!(array.validity()?, Validity::NonNullable));

    Ok(())
}

#[test]
fn validates_sparse_components() -> VortexResult<()> {
    let children = vec![
        PrimitiveArray::from_iter([10i32, 0, 30]).into_array(),
        BoolArray::from_iter([false, true, false]).into_array(),
    ];

    assert!(
        UnionArray::try_new(
            PrimitiveArray::from_iter([5u8, 9]).into_array(),
            variants()?,
            children,
        )
        .is_err()
    );
    assert!(
        UnionArray::try_new(
            PrimitiveArray::from_iter([5u8, 9, 5]).into_array(),
            variants()?,
            vec![PrimitiveArray::from_iter([10i32, 0, 30]).into_array()],
        )
        .is_err()
    );
    assert!(
        UnionArray::try_new(
            PrimitiveArray::from_iter([5u8, 9, 5]).into_array(),
            variants()?,
            vec![
                PrimitiveArray::new(buffer![10i32, 0, 30], Validity::AllValid).into_array(),
                BoolArray::from_iter([false, true, false]).into_array(),
            ],
        )
        .is_err()
    );

    Ok(())
}

#[test]
#[should_panic(expected = "Unknown UnionArray type ID 7")]
fn invalid_type_id_panics_when_accessed() {
    let array = UnionArray::try_new(
        PrimitiveArray::from_iter([5u8, 7, 5]).into_array(),
        variants().vortex_expect("valid Union variants"),
        vec![
            PrimitiveArray::from_iter([10i32, 0, 30]).into_array(),
            BoolArray::from_iter([false, true, false]).into_array(),
        ],
    )
    .vortex_expect("structurally valid UnionArray");
    let mut ctx = SESSION.create_execution_ctx();

    let _scalar = array
        .execute_scalar(1, &mut ctx)
        .vortex_expect("UnionArray scalar access");
}

#[test]
fn outer_nulls_are_independent_from_inner_nulls() -> VortexResult<()> {
    let variants = nullable_variants()?;
    let array = nullable_union_array()?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_eq!(
        array.dtype(),
        &DType::Union(variants.clone(), Nullability::Nullable)
    );
    assert_eq!(
        array.validity()?.execute_mask(array.len(), &mut ctx)?,
        Mask::from_iter([true, false, true, true])
    );
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants.clone(), 5, 10i32.into(), Nullability::Nullable)?
    );
    assert_eq!(
        array.execute_scalar(1, &mut ctx)?,
        Scalar::null(DType::Union(variants.clone(), Nullability::Nullable))
    );
    assert_eq!(
        array.execute_scalar(2, &mut ctx)?,
        Scalar::union(
            variants,
            9,
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        )?
    );

    Ok(())
}

#[test]
fn masking_adds_outer_nulls_only() -> VortexResult<()> {
    let masked = union_array()?
        .into_array()
        .mask(BoolArray::from_iter([true, false, true]).into_array())?;
    let mut ctx = SESSION.create_execution_ctx();
    let masked = masked.execute::<UnionArray>(&mut ctx)?;

    assert_eq!(
        masked.dtype(),
        &DType::Union(variants()?, Nullability::Nullable)
    );
    assert_eq!(
        masked.validity()?.execute_mask(masked.len(), &mut ctx)?,
        Mask::from_iter([true, false, true])
    );
    assert_eq!(
        masked.execute_scalar(1, &mut ctx)?,
        Scalar::null(DType::Union(variants()?, Nullability::Nullable))
    );
    assert_eq!(
        masked.execute_scalar(2, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::Nullable,)?
    );

    Ok(())
}

#[test]
fn slice_and_filter_preserve_sparse_alignment() -> VortexResult<()> {
    let array = union_array()?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let sliced = array.slice(1..3)?;
    let filtered = array.filter(Mask::from_iter([true, false, true]))?;
    assert_eq!(sliced.len(), 2);
    assert_eq!(filtered.len(), 2);
    assert_eq!(sliced.dtype(), array.dtype());
    assert_eq!(filtered.dtype(), array.dtype());
    assert_eq!(
        sliced.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?
    );
    assert_eq!(
        filtered.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?
    );

    assert_eq!(
        sliced.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable,)?
    );
    assert_eq!(
        filtered.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable,)?
    );

    Ok(())
}

/// A constant union canonicalizes into a sparse union whose selected child repeats the scalar's
/// value and whose unselected children hold placeholders.
#[test]
fn constant_union_canonicalizes_to_sparse_union() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let i32_variant = DType::Primitive(PType::I32, Nullability::NonNullable);
    let i64_variant = DType::Primitive(PType::I64, Nullability::Nullable);

    // Each case pairs a constant scalar with the dtype of a variant it does not select.
    let cases = [
        (
            Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?,
            i32_variant.clone(),
        ),
        (
            Scalar::union(nullable_variants()?, 5, 7i32.into(), Nullability::Nullable)?,
            i64_variant,
        ),
        (
            Scalar::null(DType::Union(variants()?, Nullability::Nullable)),
            i32_variant,
        ),
    ];

    for (scalar, unselected_dtype) in cases {
        let canonical = ConstantArray::new(scalar.clone(), 3)
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_union();

        assert_eq!(canonical.dtype(), scalar.dtype());
        assert_eq!(canonical.len(), 3);

        let unselected = canonical
            .iter_children()
            .find(|child| child.dtype() == &unselected_dtype)
            .ok_or_else(|| vortex_err!("No child with dtype {unselected_dtype}"))?;
        assert_eq!(
            unselected.execute_scalar(0, &mut ctx)?,
            Scalar::default_value(&unselected_dtype)
        );

        for index in 0..canonical.len() {
            assert_eq!(canonical.execute_scalar(index, &mut ctx)?, scalar);
        }
    }

    Ok(())
}

#[test]
fn constant_union_reports_uncompressed_size() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let scalar = Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?;

    // Four `u8` type IDs, four `i32` rows, and four `bool` placeholder bits rounded up to a byte.
    let size = uncompressed_size_in_bytes(&ConstantArray::new(scalar, 4).into_array(), &mut ctx)?;

    assert_eq!(size, 4 + 4 * 4 + 1);

    Ok(())
}

/// Serialize `array` into its wire form and the context that resolves its encoding IDs.
fn serialize(array: &UnionArray) -> VortexResult<(SerializedArray, ReadContext)> {
    let array_ctx = ArrayContext::empty();
    let mut concat = ByteBufferMut::empty();
    for buffer in
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())?
    {
        concat.extend_from_slice(buffer.as_ref());
    }
    Ok((
        SerializedArray::try_from(concat.freeze())?,
        ReadContext::new(array_ctx.to_ids()),
    ))
}

/// Assert that `actual` is a union holding the same rows as `expected`.
///
/// This compares row by row because `assert_arrays_eq!` cannot compare union arrays yet.
#[track_caller]
fn assert_same_rows(actual: &ArrayRef, expected: &UnionArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.as_::<Union>().variants(), expected.variants());
    for index in 0..expected.len() {
        assert_eq!(
            actual.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}

/// The `variants` schema with a third variant, so it expects one more child.
fn variants_with_extra() -> VortexResult<UnionVariants> {
    UnionVariants::try_new(
        ["number", "flag", "extra"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
            DType::Primitive(PType::U16, Nullability::NonNullable),
        ],
        vec![5, 9, 11],
    )
}

#[test]
fn serde_roundtrip() -> VortexResult<()> {
    for array in [union_array()?, nullable_union_array()?] {
        let (serialized, read_ctx) = serialize(&array)?;
        let decoded = serialized.decode(array.dtype(), array.len(), &read_ctx, &SESSION)?;
        assert_same_rows(&decoded, &array)?;
    }

    Ok(())
}

#[test]
fn deserialize_rejects_malformed_parts() -> VortexResult<()> {
    let array = union_array()?;
    let dtype = array.dtype().clone();
    let len = array.len();
    let UnionDataParts {
        type_ids, children, ..
    } = array.clone().into_data_parts();
    let mut all = vec![type_ids.clone()];
    all.extend(children);
    let deserialize =
        |dtype: &DType, metadata: &[u8], buffers: &[BufferHandle], children: &[ArrayRef]| {
            Union.deserialize(
                ArrayDeserialization::new(
                    ArrayPlugin::id(&Union),
                    dtype,
                    len,
                    metadata,
                    buffers,
                    &children,
                ),
                &SESSION,
            )
        };

    let restored = deserialize(&dtype, &[], &[], &all)?;
    assert_same_rows(&restored, &array)?;

    // Readers ignore children that follow the type IDs and one child per variant.
    let mut with_trailing = all.clone();
    with_trailing.push(type_ids);
    let restored = deserialize(&dtype, &[], &[], &with_trailing)?;
    assert_same_rows(&restored, &array)?;

    // Too few children for the variants the dtype declares.
    for count in 0..all.len() {
        assert!(deserialize(&dtype, &[], &[], &all[..count]).is_err());
    }
    let extra = DType::Union(variants_with_extra()?, Nullability::NonNullable);
    assert!(deserialize(&extra, &[], &[], &all).is_err());

    // A dtype that is not a union.
    let not_union = DType::Primitive(PType::U8, Nullability::NonNullable);
    assert!(deserialize(&not_union, &[], &[], &all).is_err());

    // Union metadata must be a well-formed protobuf message, and union arrays have no buffers.
    assert!(deserialize(&dtype, &[0], &[], &all).is_err());
    let buffer = BufferHandle::new_host(buffer![0u8]);
    assert!(deserialize(&dtype, &[], &[buffer], &all).is_err());

    Ok(())
}

/// Decoding the wire form against a dtype or length it was not written with returns an error.
#[test]
fn decode_rejects_mismatched_dtype_and_len() -> VortexResult<()> {
    let array = union_array()?;
    let len = array.len();
    let (serialized, read_ctx) = serialize(&array)?;

    assert!(
        serialized
            .decode(array.dtype(), len + 1, &read_ctx, &SESSION)
            .is_err()
    );

    let wider_variant = UnionVariants::try_new(
        ["number", "flag"].into(),
        vec![
            DType::Primitive(PType::I64, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
        vec![5, 9],
    )?;
    for dtype in [
        DType::Union(wider_variant, Nullability::NonNullable),
        DType::Union(variants_with_extra()?, Nullability::NonNullable),
        DType::Primitive(PType::U8, Nullability::NonNullable),
    ] {
        assert!(serialized.decode(&dtype, len, &read_ctx, &SESSION).is_err());
    }

    Ok(())
}
