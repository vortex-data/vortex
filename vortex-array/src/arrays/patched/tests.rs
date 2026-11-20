// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn indices(positions: &[u32]) -> ArrayRef {
    PrimitiveArray::from_iter(positions.iter().copied()).into_array()
}

/// A reference result: `base` with `patches` applied.
fn reference<T: Clone>(base: Vec<T>, patches: &[(usize, T)]) -> Vec<T> {
    let mut out = base;
    for (pos, value) in patches {
        out[*pos] = value.clone();
    }
    out
}

#[rstest]
#[case::one_lane(1)]
#[case::fastlanes_lanes(32)]
#[case::many_lanes(1024)]
fn nullable_patches_over_primitive(#[case] n_lanes: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 3000u32;
    let inner = PrimitiveArray::from_option_iter((0..len).map(|i| (i % 7 != 0).then_some(i)));
    let positions = [0u32, 7, 8, 1023, 1024, 2047, 2999];
    let values = PrimitiveArray::from_option_iter([
        Some(100u32),
        Some(101),
        None,
        Some(103),
        None,
        Some(105),
        Some(106),
    ]);
    let patched = Patched::try_new_with_lanes(
        inner.into_array(),
        indices(&positions),
        values.into_array(),
        n_lanes,
        &mut ctx,
    )?
    .into_array();

    let base: Vec<Option<u32>> = (0..len).map(|i| (i % 7 != 0).then_some(i)).collect();
    let expected = reference(
        base,
        &[
            (0, Some(100)),
            (7, Some(101)),
            (8, None),
            (1023, Some(103)),
            (1024, None),
            (2047, Some(105)),
            (2999, Some(106)),
        ],
    );
    assert_arrays_eq!(
        patched.clone(),
        PrimitiveArray::from_option_iter(expected.clone()),
        &mut ctx
    );

    // Slices at unaligned offsets skip the patches outside the slice.
    for (start, end) in [(1, 3000), (8, 1025), (1024, 2048), (1500, 1600)] {
        assert_arrays_eq!(
            patched.slice(start..end)?,
            PrimitiveArray::from_option_iter(expected[start..end].iter().copied()),
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn validity_is_patched_lazily() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patched = Patched::sparse(
        indices(&[1, 3]),
        buffer![10i32, 30].into_array(),
        5,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        &mut ctx,
    )?
    .into_array();

    let Validity::Array(validity) = patched.validity()? else {
        panic!("expected a validity array");
    };
    assert!(validity.is::<Patched>());
    assert_arrays_eq!(
        validity,
        BoolArray::from_iter([false, true, false, true, false]),
        &mut ctx
    );
    assert_arrays_eq!(
        patched,
        PrimitiveArray::from_option_iter([None, Some(10i32), None, Some(30), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn sparse_strings() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = VarBinViewArray::from_iter_nullable_str([
        Some("a patch value that is too long to inline"),
        None,
        Some("short"),
    ]);
    let patched = Patched::sparse(
        indices(&[2, 1500, 1501]),
        values.into_array(),
        2000,
        Scalar::utf8("a fill value that is too long to inline", Nullability::Nullable),
        &mut ctx,
    )?
    .into_array();

    let fill = Some("a fill value that is too long to inline");
    let expected = reference(
        vec![fill; 2000],
        &[
            (2, Some("a patch value that is too long to inline")),
            (1500, None),
            (1501, Some("short")),
        ],
    );
    assert_arrays_eq!(
        patched.clone(),
        VarBinViewArray::from_iter_nullable_str(expected.clone()),
        &mut ctx
    );

    let taken = patched.take(buffer![1501u32, 0, 2, 1500].into_array())?;
    assert_arrays_eq!(
        taken,
        VarBinViewArray::from_iter_nullable_str([expected[1501], fill, expected[2], None]),
        &mut ctx
    );

    let mask = Mask::from_indices(2000, vec![1, 2, 1500, 1999]);
    let filtered = patched.filter(mask)?;
    assert_arrays_eq!(
        filtered,
        VarBinViewArray::from_iter_nullable_str([fill, expected[2], None, fill]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn sparse_bools() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patched = Patched::sparse(
        indices(&[0, 5, 9]),
        BoolArray::from_iter([Some(true), None, Some(false)]).into_array(),
        10,
        Scalar::bool(false, Nullability::NonNullable),
        &mut ctx,
    )?
    .into_array();
    let mut expected = vec![Some(false); 10];
    expected[0] = Some(true);
    expected[5] = None;
    assert_arrays_eq!(patched, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn decimals() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let inner = DecimalArray::from_iter([1i64, 2, 3, 4], dtype);
    let values = DecimalArray::from_iter([20i64, 40], dtype);
    let patched =
        Patched::try_new(inner.into_array(), indices(&[1, 3]), values.into_array(), &mut ctx)?;
    assert_arrays_eq!(
        patched,
        DecimalArray::from_iter([1i64, 20, 3, 40], dtype),
        &mut ctx
    );
    Ok(())
}

#[test]
fn structs_patch_each_field() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = StructArray::from_fields(&[
        ("a", buffer![1i32, 2, 3].into_array()),
        (
            "b",
            VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array(),
        ),
    ])?;
    let values = StructArray::from_fields(&[
        ("a", buffer![20i32].into_array()),
        ("b", VarBinViewArray::from_iter_str(["patched"]).into_array()),
    ])?;
    let patched =
        Patched::try_new(inner.into_array(), indices(&[1]), values.into_array(), &mut ctx)?;
    let expected = StructArray::from_fields(&[
        ("a", buffer![1i32, 20, 3].into_array()),
        (
            "b",
            VarBinViewArray::from_iter_str(["x", "patched", "z"]).into_array(),
        ),
    ])?;
    assert_arrays_eq!(patched, expected, &mut ctx);
    Ok(())
}

#[test]
fn fixed_size_lists_use_take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = FixedSizeListArray::new(
        buffer![1i32, 1, 2, 2, 3, 3].into_array(),
        2,
        Validity::NonNullable,
        3,
    );
    let values =
        FixedSizeListArray::new(buffer![9i32, 9].into_array(), 2, Validity::NonNullable, 1);
    let patched =
        Patched::try_new(inner.into_array(), indices(&[2]), values.into_array(), &mut ctx)?;
    let expected = FixedSizeListArray::new(
        buffer![1i32, 1, 2, 2, 9, 9].into_array(),
        2,
        Validity::NonNullable,
        3,
    );
    assert_arrays_eq!(patched, expected, &mut ctx);
    Ok(())
}

#[test]
fn compare_pushes_down_into_children() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = PrimitiveArray::from_iter(0u32..2048);
    let values = PrimitiveArray::from_option_iter([Some(7u32), None, Some(7)]);
    let patched = Patched::try_new(
        inner.into_array(),
        indices(&[1, 2, 1500]),
        values.into_array(),
        &mut ctx,
    )?
    .into_array();

    let compared = patched
        .binary(
            ConstantArray::new(7u32, patched.len()).into_array(),
            Operator::Eq,
        )?
        .optimize_ctx(&SESSION)?;
    assert!(compared.is::<Patched>(), "{}", compared.display_tree());

    let expected = (0u32..2048).map(|i| match i {
        1 | 1500 => Some(true),
        2 => None,
        i => Some(i == 7),
    });
    assert_arrays_eq!(compared, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn widening_cast_pushes_down() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patched = Patched::try_new(
        buffer![1u8, 2, 3].into_array(),
        indices(&[1]),
        buffer![200u8].into_array(),
        &mut ctx,
    )?
    .into_array();
    let cast = patched
        .cast(DType::Primitive(PType::U32, Nullability::Nullable))?
        .optimize_ctx(&SESSION)?;
    assert!(cast.is::<Patched>(), "{}", cast.display_tree());
    assert_arrays_eq!(
        cast,
        PrimitiveArray::from_option_iter([Some(1u32), Some(200), Some(3)]),
        &mut ctx
    );

    // A narrowing cast is not pushed down, but still evaluates.
    let narrowed = patched
        .cast(DType::Primitive(PType::I8, Nullability::NonNullable))?
        .optimize_ctx(&SESSION)?;
    assert!(!narrowed.is::<Patched>());
    assert!(narrowed.execute::<crate::Canonical>(&mut ctx).is_err());
    Ok(())
}

#[test]
fn take_and_filter_primitive_across_chunks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 5000usize;
    let positions: Vec<u32> = (0..len as u32).step_by(37).collect();
    let values = PrimitiveArray::from_iter(positions.iter().map(|p| p + 1_000_000));
    let patched = Patched::try_new(
        PrimitiveArray::from_iter(0..len as u32).into_array(),
        indices(&positions),
        values.into_array(),
        &mut ctx,
    )?
    .into_array()
    .slice(3..4900)?;
    let expected: Vec<u32> = (3..4900u32)
        .map(|i| if i % 37 == 0 { i + 1_000_000 } else { i })
        .collect();

    let take_indices: Vec<u64> = (0..expected.len() as u64).rev().step_by(3).collect();
    let taken = patched.take(PrimitiveArray::from_iter(take_indices.clone()).into_array())?;
    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_iter(take_indices.iter().map(|&i| expected[i as usize])),
        &mut ctx
    );

    let mask = Mask::from_iter((0..expected.len()).map(|i| i % 5 == 0 || i % 37 == 34));
    let filtered = patched.filter(mask.clone())?;
    assert_arrays_eq!(
        filtered,
        PrimitiveArray::from_iter(
            expected
                .iter()
                .enumerate()
                .filter(|(i, _)| mask.value(*i))
                .map(|(_, v)| *v)
        ),
        &mut ctx
    );
    Ok(())
}
