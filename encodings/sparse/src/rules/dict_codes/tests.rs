// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Sparse;

#[rstest]
#[case([Some(0u8), Some(0), Some(1)])]
#[case([Some(1u8), None, Some(0)])]
#[case([None, None, None])]
fn repeated_and_null_codes(
    #[case] patch_codes: [Option<u8>; 3],
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let codes = Sparse::try_new(
        buffer![1u32, 3, 6].into_array(),
        PrimitiveArray::from_option_iter(patch_codes).into_array(),
        8,
        Scalar::null_native::<u8>(),
        &mut ctx,
    )?
    .into_array();

    let range = if sliced { 1..7 } else { 0..8 };
    let codes = codes.slice(range.clone())?;
    let values = PrimitiveArray::from_option_iter([Some(10i32), None]).into_array();
    let dict = DictArray::try_new(codes, values)?.into_array();

    let reduced = if sliced {
        dict.clone().execute_until::<Sparse>(&mut ctx)?
    } else {
        dict.optimize()?
    };
    assert!(reduced.is::<Sparse>(), "{}", reduced.display_tree());

    let mut expected = [None; 8];
    for (position, code) in [1, 3, 6].into_iter().zip(patch_codes) {
        expected[position] = code.and_then(|code| [Some(10), None][usize::from(code)]);
    }
    let expected = PrimitiveArray::from_option_iter(expected[range].iter().copied());

    assert_arrays_eq!(reduced, expected, &mut ctx);
    assert_arrays_eq!(dict, expected, &mut ctx);
    Ok(())
}

#[test]
fn sparse_struct_uses_empty_nonnullable_list_defaults() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let list = ListArray::try_new(
        buffer![7i32, 8, 9].into_array(),
        buffer![0u32, 2, 3].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    let values = StructArray::from_fields(&[
        ("list", list.into_array()),
        (
            "id",
            PrimitiveArray::from_option_iter([Some(10i32), None]).into_array(),
        ),
    ])?;

    let codes = Sparse::try_new(
        buffer![1u32, 3].into_array(),
        PrimitiveArray::new(buffer![0u8, 1], Validity::AllValid).into_array(),
        5,
        Scalar::null_native::<u8>(),
        &mut ctx,
    )?;
    let dict = DictArray::try_new(codes.into_array(), values.into_array())?.into_array();

    let reduced = dict.optimize()?;
    assert!(reduced.is::<Sparse>(), "{}", reduced.display_tree());
    let actual = reduced.execute::<StructArray>(&mut ctx)?;
    assert!(actual.unmasked_field(0).is::<Sparse>());

    // Null struct rows still need valid empty defaults in their non-nullable list child.
    let expected_lists = ListArray::try_new(
        buffer![7i32, 8, 9].into_array(),
        buffer![0u32, 0, 2, 2, 3, 3].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;

    assert_arrays_eq!(actual.unmasked_field(0), expected_lists, &mut ctx);
    assert_arrays_eq!(
        actual,
        StructArray::try_from_iter_with_validity(
            [
                ("list", expected_lists.into_array()),
                (
                    "id",
                    PrimitiveArray::from_option_iter([None, Some(10i32), None, None, None])
                        .into_array()
                ),
            ],
            Validity::from_iter([false, true, false, true, false]),
        )?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn nonnull_fill_keeps_dictionary() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let codes = Sparse::try_new(
        buffer![2u32].into_array(),
        PrimitiveArray::new(buffer![1u8], Validity::AllValid).into_array(),
        4,
        Scalar::primitive(0u8, Nullability::Nullable),
        &mut ctx,
    )?;
    let dict =
        DictArray::try_new(codes.into_array(), buffer![10i32, 20].into_array())?.into_array();

    assert!(dict.optimize()?.is::<vortex_array::arrays::Dict>());
    assert_arrays_eq!(
        dict,
        PrimitiveArray::from_option_iter([Some(10i32), Some(10), Some(20), Some(10)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn unsupported_sparse_type_is_not_rewritten() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let codes = Sparse::try_new(
        buffer![1u32].into_array(),
        PrimitiveArray::new(buffer![0u8], Validity::AllValid).into_array(),
        3,
        Scalar::null_native::<u8>(),
        &mut ctx,
    )?
    .into_array();

    let values =
        ConstantArray::new(Scalar::null(DType::Variant(Nullability::Nullable)), 1).into_array();
    let dict = DictArray::try_new(codes.clone(), values)?.into_array();

    assert!(codes.reduce_parent(&dict, 0)?.is_none());
    Ok(())
}
