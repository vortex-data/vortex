// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
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
    let mut ctx = array_session().create_execution_ctx();

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
    let mut ctx = array_session().create_execution_ctx();

    let _scalar = array
        .execute_scalar(1, &mut ctx)
        .vortex_expect("UnionArray scalar access");
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
fn slice_and_filter_preserve_sparse_alignment() -> VortexResult<()> {
    let array = union_array()?.into_array();
    let mut ctx = array_session().create_execution_ctx();

    let sliced = array.slice(1..3)?;
    let filtered = array.filter(Mask::from_iter([true, false, true]))?;

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

#[test]
fn constant_union_canonicalizes_to_sparse_union() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Union(variants()?, Nullability::Nullable);

    for scalar in [
        Scalar::union(variants()?, 9, true.into(), Nullability::Nullable)?,
        Scalar::null(dtype.clone()),
    ] {
        let canonical = ConstantArray::new(scalar.clone(), 3)
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_union();

        assert_eq!(canonical.dtype(), &dtype);

        // The unselected variant is only a placeholder, so it keeps its declared dtype.
        assert_eq!(
            canonical.child_by_name("number")?.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        for index in 0..canonical.len() {
            assert_eq!(canonical.execute_scalar(index, &mut ctx)?, scalar);
        }
    }

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
