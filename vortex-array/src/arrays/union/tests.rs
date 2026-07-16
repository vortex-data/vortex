// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::Canonical;
use crate::CanonicalValidity;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::union::UnionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnionVariants;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

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
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        array.child_by_name("flag")?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    assert!(array.child_by_name_opt("missing").is_none());
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable,)?
    );
    assert_eq!(
        array.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable,)?
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

    let unknown_type_id = UnionArray::try_new(
        PrimitiveArray::from_iter([5u8, 7, 5]).into_array(),
        variants()?,
        children.clone(),
    );
    assert!(unknown_type_id.is_err());

    let mismatched_lengths = UnionArray::try_new(
        PrimitiveArray::from_iter([5u8, 9]).into_array(),
        variants()?,
        children,
    );
    assert!(mismatched_lengths.is_err());

    let nullable_child = UnionArray::try_new(
        PrimitiveArray::from_iter([5u8, 9, 5]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::new(buffer![10i32, 0, 30], Validity::AllValid).into_array(),
            BoolArray::from_iter([false, true, false]).into_array(),
        ],
    );
    assert!(nullable_child.is_err());

    Ok(())
}

#[test]
fn outer_nulls_are_independent_from_inner_nulls() -> VortexResult<()> {
    let variants = nullable_variants()?;
    let array = nullable_union_array()?;
    let mut ctx = array_session().create_execution_ctx();

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
        Scalar::union(variants.clone(), 5, 10i32.into(), Nullability::Nullable,)?
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
    let mut ctx = array_session().create_execution_ctx();
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
fn structural_operations_preserve_sparse_alignment() -> VortexResult<()> {
    let array = union_array()?.into_array();
    let mut ctx = array_session().create_execution_ctx();

    let sliced = array.slice(1..3)?;
    let filtered = array.filter(Mask::from_iter([true, false, true]))?;
    let taken = array.take(PrimitiveArray::from_iter([2u32, 1]).into_array())?;

    assert_eq!(
        sliced.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable,)?
    );
    assert_eq!(
        filtered.execute_scalar(1, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable,)?
    );
    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable,)?
    );

    Ok(())
}

#[test]
fn serde_roundtrip() -> VortexResult<()> {
    let session = array_session();
    let mut execution_ctx = session.create_execution_ctx();
    for array in [union_array()?, nullable_union_array()?] {
        let dtype = array.dtype().clone();
        let len = array.len();
        let array_ctx = ArrayContext::empty();
        let serialized = array.clone().into_array().serialize(
            &array_ctx,
            &session,
            &SerializeOptions::default(),
        )?;
        let mut concat = ByteBufferMut::empty();
        for buffer in serialized {
            concat.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(concat.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_ctx.to_ids()),
            &session,
        )?;
        let decoded = decoded.as_::<Union>();

        assert_eq!(decoded.variants(), array.variants());
        for index in 0..len {
            assert_eq!(
                decoded.array().execute_scalar(index, &mut execution_ctx)?,
                array.execute_scalar(index, &mut execution_ctx)?
            );
        }
    }

    Ok(())
}

#[test]
fn constant_union_executes_to_sparse_union() -> VortexResult<()> {
    let scalar = Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?;
    let mut ctx = array_session().create_execution_ctx();
    let array = ConstantArray::new(scalar.clone(), 3)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;

    assert_eq!(array.len(), 3);
    assert_eq!(array.execute_scalar(2, &mut ctx)?, scalar);

    let variants = nullable_variants()?;
    let inner_null = Scalar::union(
        variants.clone(),
        9,
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        Nullability::Nullable,
    )?;
    let array = ConstantArray::new(inner_null.clone(), 3)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;
    assert_eq!(
        array.validity()?.execute_mask(array.len(), &mut ctx)?,
        Mask::new_true(3)
    );
    assert_eq!(array.execute_scalar(1, &mut ctx)?, inner_null);

    let outer_null = Scalar::null(DType::Union(variants, Nullability::Nullable));
    let array = ConstantArray::new(outer_null.clone(), 3)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;
    assert_eq!(
        array.validity()?.execute_mask(array.len(), &mut ctx)?,
        Mask::new_false(3)
    );
    assert_eq!(array.execute_scalar(1, &mut ctx)?, outer_null);

    Ok(())
}

#[test]
fn constant_union_builds_nested_union_placeholders() -> VortexResult<()> {
    let nested_variants = UnionVariants::try_new(
        ["value"].into(),
        vec![DType::Primitive(PType::I64, Nullability::NonNullable)],
        vec![3],
    )?;
    let nested_dtype = DType::Union(nested_variants, Nullability::NonNullable);
    let outer_variants = UnionVariants::try_new(
        ["number", "nested"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            nested_dtype.clone(),
        ],
        vec![5, 9],
    )?;
    let mut ctx = array_session().create_execution_ctx();

    let selected_number = Scalar::union(
        outer_variants.clone(),
        5,
        42i32.into(),
        Nullability::NonNullable,
    )?;
    let array = ConstantArray::new(selected_number, 2)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;
    assert_eq!(
        array.child_by_name("nested")?.execute_scalar(0, &mut ctx)?,
        Scalar::zero_value(&nested_dtype)
    );

    let outer_null = Scalar::null(DType::Union(outer_variants, Nullability::Nullable));
    let array = ConstantArray::new(outer_null, 2)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;
    assert_eq!(
        array.child_by_name("nested")?.execute_scalar(0, &mut ctx)?,
        Scalar::zero_value(&nested_dtype)
    );

    Ok(())
}

#[test]
fn constant_union_builds_deeply_nested_union_placeholders() -> VortexResult<()> {
    let nested_dtype = DType::Union(
        UnionVariants::try_new(
            ["value"].into(),
            vec![DType::Primitive(PType::I64, Nullability::NonNullable)],
            vec![3],
        )?,
        Nullability::NonNullable,
    );
    let wrapper_dtype =
        DType::struct_([("nested", nested_dtype.clone())], Nullability::NonNullable);
    let outer_variants = UnionVariants::try_new(
        ["number", "wrapper"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            wrapper_dtype,
        ],
        vec![5, 9],
    )?;
    let selected_number =
        Scalar::union(outer_variants, 5, 42_i32.into(), Nullability::NonNullable)?;
    let mut ctx = array_session().create_execution_ctx();
    let array = ConstantArray::new(selected_number, 2)
        .into_array()
        .execute::<UnionArray>(&mut ctx)?;
    let wrapper = array
        .child_by_name("wrapper")?
        .execute_scalar(0, &mut ctx)?;

    assert_eq!(
        wrapper.as_struct().field("nested"),
        Some(Scalar::default_value(&nested_dtype))
    );

    Ok(())
}

#[test]
fn constant_union_rejects_uninhabited_placeholders_without_panicking() -> VortexResult<()> {
    let uninhabited = DType::Union(
        UnionVariants::new(Default::default(), vec![])?,
        Nullability::NonNullable,
    );
    let outer_variants = UnionVariants::try_new(
        ["number", "uninhabited"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            uninhabited,
        ],
        vec![5, 9],
    )?;
    let selected_number =
        Scalar::union(outer_variants, 5, 42_i32.into(), Nullability::NonNullable)?;
    let mut ctx = array_session().create_execution_ctx();

    assert!(
        ConstantArray::new(selected_number.clone(), 1)
            .into_array()
            .execute::<UnionArray>(&mut ctx)
            .is_err()
    );
    assert_eq!(
        ConstantArray::new(selected_number, 0)
            .into_array()
            .execute::<UnionArray>(&mut ctx)?
            .len(),
        0
    );

    Ok(())
}

#[test]
fn empty_union_supports_variant_children() -> VortexResult<()> {
    let variants = UnionVariants::try_new(
        ["dynamic"].into(),
        vec![DType::Variant(Nullability::NonNullable)],
        vec![5],
    )?;
    let array = Canonical::empty(&DType::Union(variants, Nullability::NonNullable)).into_union();

    assert_eq!(array.len(), 0);
    assert_eq!(
        array.child(0).map(|child| child.dtype()),
        Some(&DType::Variant(Nullability::NonNullable))
    );

    Ok(())
}

#[test]
fn canonical_validity_canonicalizes_union_type_ids() -> VortexResult<()> {
    let array = UnionArray::try_new(
        ConstantArray::new(Scalar::primitive(5_u8, Nullability::Nullable), 2).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10_i32, 20]).into_array(),
            BoolArray::from_iter([false, true]).into_array(),
        ],
    )?;
    let mut ctx = array_session().create_execution_ctx();
    let Canonical::Union(array) = array.into_array().execute::<CanonicalValidity>(&mut ctx)?.0
    else {
        return Err(vortex_err!(
            "UnionArray must remain canonical Union storage"
        ));
    };

    assert!(array.type_ids().is::<crate::arrays::Primitive>());
    assert_eq!(array.dtype().nullability(), Nullability::Nullable);

    Ok(())
}

#[test]
fn chunked_union_packs_components() -> VortexResult<()> {
    let first = union_array()?.into_array().slice(0..1)?;
    let second = union_array()?.into_array().slice(1..3)?;
    let dtype = first.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![first, second], dtype)?.into_array();
    let mut ctx = array_session().create_execution_ctx();
    let canonical = chunked.execute::<UnionArray>(&mut ctx)?;

    assert!(canonical.type_ids().is::<Chunked>());
    assert!(canonical.iter_children().all(|child| child.is::<Chunked>()));
    assert_eq!(
        canonical.execute_scalar(2, &mut ctx)?,
        Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable,)?
    );

    Ok(())
}
