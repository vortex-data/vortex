// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

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
        patched,
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
        Scalar::utf8(
            "a fill value that is too long to inline",
            Nullability::Nullable,
        ),
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
        patched,
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
    let patched = Patched::try_new(
        inner.into_array(),
        indices(&[1, 3]),
        values.into_array(),
        &mut ctx,
    )?;
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
        (
            "b",
            VarBinViewArray::from_iter_str(["patched"]).into_array(),
        ),
    ])?;
    let patched = Patched::try_new(
        inner.into_array(),
        indices(&[1]),
        values.into_array(),
        &mut ctx,
    )?;
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
    let patched = Patched::try_new(
        inner.into_array(),
        indices(&[2]),
        values.into_array(),
        &mut ctx,
    )?;
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

mod serde {
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::ArrayContext;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySessionExt;

    /// Serialize and decode `array`, checking that it decodes to a patched array of the same
    /// values, and returning the decoded array.
    fn round_trip(array: ArrayRef) -> VortexResult<ArrayRef> {
        let session = VortexSession::empty();
        session.arrays().register(Patched);
        let ctx = ArrayContext::empty().with_allowed_ids(
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect()),
        );
        let mut bytes = ByteBufferMut::empty();
        for buffer in array.serialize(&ctx, &session, &SerializeOptions::default())? {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(ctx.to_ids()),
            &session,
        )?;
        assert!(decoded.is::<Patched>(), "{}", decoded.display_tree());
        let mut exec = session.create_execution_ctx();
        let expected = array.execute::<crate::Canonical>(&mut exec)?.into_array();
        assert_arrays_eq!(decoded, expected, &mut exec);
        Ok(decoded)
    }

    #[test]
    fn nullable_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Patched::try_new(
            PrimitiveArray::from_option_iter((0..3000u32).map(|i| (i % 3 != 0).then_some(i)))
                .into_array(),
            indices(&[0, 5, 1500, 2999]),
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), None]).into_array(),
            &mut ctx,
        )?;
        round_trip(array.into_array())?;
        Ok(())
    }

    #[test]
    fn sliced_arrays_write_only_their_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let positions: Vec<u32> = (0..5000).step_by(10).collect();
        let array = Patched::try_new(
            PrimitiveArray::from_iter(0..5000u32).into_array(),
            indices(&positions),
            PrimitiveArray::from_iter(positions.iter().map(|p| p + 100_000)).into_array(),
            &mut ctx,
        )?
        .into_array();

        let sliced = array.slice(2100..3500)?.optimize_ctx(&SESSION)?;
        assert_eq!(sliced.as_::<Patched>().patch_values().len(), 500);
        let decoded = round_trip(sliced)?;
        // Chunks 2 and 3 hold positions 2048..4096, with one patch every 10 positions.
        let decoded = decoded.as_::<Patched>();
        assert_eq!(
            decoded.patch_values().len(),
            (2048..4096).filter(|p| p % 10 == 0).count()
        );

        // A slice within one chunk, at an offset.
        round_trip(array.slice(1030..1031)?.optimize_ctx(&SESSION)?)?;
        Ok(())
    }

    #[test]
    fn sparse_strings_and_bools() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let strings = Patched::sparse(
            indices(&[1, 1500]),
            VarBinViewArray::from_iter_nullable_str([
                Some("a long patch value, not inlined"),
                None,
            ])
            .into_array(),
            2000,
            Scalar::utf8("fill", Nullability::NonNullable),
            &mut ctx,
        )?;
        round_trip(strings.into_array())?;

        let bools = Patched::sparse_with_lanes(
            indices(&[0, 7]),
            BoolArray::from_iter([true, true]).into_array(),
            10,
            Scalar::bool(false, Nullability::NonNullable),
            4,
            &mut ctx,
        )?;
        round_trip(bools.into_array())?;
        Ok(())
    }
}

mod conformance {
    use std::f32;

    use super::*;
    use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::compute::conformance::take::test_take_conformance;

    fn sparse(positions: &[u32], values: ArrayRef, len: usize, fill: Scalar) -> ArrayRef {
        Patched::sparse(
            indices(positions),
            values,
            len,
            fill,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    }

    fn over(inner: ArrayRef, positions: &[u32], values: ArrayRef) -> ArrayRef {
        Patched::try_new(
            inner,
            indices(positions),
            values,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    }

    fn arrays() -> Vec<ArrayRef> {
        vec![
            sparse(
                &[2, 5, 8],
                PrimitiveArray::from_option_iter([Some(100i32), Some(200), Some(300)]).into_array(),
                10,
                Scalar::null_native::<i32>(),
            ),
            sparse(
                &[1, 3, 7],
                buffer![42i32, 84, 126].into_array(),
                10,
                Scalar::from(0i32),
            ),
            sparse(
                &[0, 4, 9],
                buffer![1000u64, 2000, 3000].into_array(),
                10,
                Scalar::from(999u64),
            ),
            sparse(
                &[2, 6],
                buffer![f32::consts::PI, f32::consts::E].into_array(),
                8,
                Scalar::from(0.0f32),
            ),
            sparse(
                &[0, 1, 2, 3, 4],
                PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, Some(40), Some(50)])
                    .into_array(),
                5,
                Scalar::null_native::<i32>(),
            ),
            sparse(
                &[100, 500, 900, 1500, 1999],
                buffer![111i32, 222, 333, 444, 555].into_array(),
                2000,
                Scalar::from(0i32),
            ),
            // A non-constant inner array with nulls, patched across chunks, and then sliced.
            over(
                PrimitiveArray::from_option_iter((0..3000i64).map(|i| (i % 11 != 0).then_some(i)))
                    .into_array(),
                &[0, 11, 1023, 1024, 2047, 2999],
                PrimitiveArray::from_option_iter([
                    Some(-1i64),
                    Some(-2),
                    None,
                    Some(-4),
                    Some(-5),
                    Some(-6),
                ])
                .into_array(),
            )
            .slice(5..2500)
            .unwrap()
            .optimize_ctx(&SESSION)
            .unwrap(),
            sparse(
                &[1, 4, 7],
                VarBinViewArray::from_iter_nullable_str([
                    Some("a patch long enough not to be inlined"),
                    None,
                    Some("b"),
                ])
                .into_array(),
                10,
                Scalar::utf8("fill", Nullability::NonNullable),
            ),
            sparse(
                &[0, 3],
                BoolArray::from_iter([Some(true), None]).into_array(),
                6,
                Scalar::bool(false, Nullability::NonNullable),
            ),
        ]
    }

    #[test]
    fn consistency() {
        for array in arrays() {
            let mut ctx = SESSION.create_execution_ctx();
            test_array_consistency(&array, &mut ctx);
            test_filter_conformance(&array, &mut ctx);
            test_take_conformance(&array, &mut ctx);
            test_mask_conformance(&array, &mut ctx);
            test_cast_conformance(&array, &mut ctx);
        }
    }

    #[test]
    fn binary_numeric() {
        for array in arrays() {
            if array.dtype().is_primitive() {
                test_binary_numeric_array(&array, &mut SESSION.create_execution_ctx());
            }
        }
    }
}

mod aggregates {
    use super::*;
    use crate::aggregate_fn::AggregateFn;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::nan_count::NanCount;
    use crate::aggregate_fn::fns::null_count::NullCount;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::arrays::patched::compute::aggregate::PatchedConstantAggregate;

    /// Aggregate `array` with the kernel, and its canonical form without it.
    fn check<V: AggregateFnVTable>(vtable: V, options: V::Options, array: &ArrayRef)
    where
        V::Options: Clone,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let aggregate_fn = AggregateFn::new(vtable, options).erased();
        let kernel = PatchedConstantAggregate
            .aggregate(&aggregate_fn, array, &mut ctx)
            .unwrap()
            .expect("kernel applies to a constant inner array");

        let canonical = array
            .clone()
            .execute::<crate::Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        let mut accumulator = aggregate_fn.accumulator(array.dtype()).unwrap();
        accumulator.accumulate(&canonical, &mut ctx).unwrap();
        let mut from_kernel = aggregate_fn.accumulator(array.dtype()).unwrap();
        from_kernel.combine_partials(&kernel).unwrap();
        assert_eq!(
            from_kernel.final_scalar().unwrap(),
            accumulator.final_scalar().unwrap(),
            "{aggregate_fn}"
        );
    }

    #[rstest]
    #[case::null_fill(Patched::sparse(
        indices(&[2, 5, 1500]),
        PrimitiveArray::from_option_iter([Some(7i64), None, Some(-3)]).into_array(),
        2000,
        Scalar::null_native::<i64>(),
        &mut SESSION.create_execution_ctx(),
    ).unwrap().into_array())]
    #[case::value_fill(Patched::sparse(
        indices(&[0, 3]),
        buffer![10i64, 20].into_array(),
        5,
        Scalar::from(4i64),
        &mut SESSION.create_execution_ctx(),
    ).unwrap().into_array())]
    #[case::fully_patched(Patched::sparse(
        indices(&[0, 1]),
        buffer![1i64, 1].into_array(),
        2,
        Scalar::from(100i64),
        &mut SESSION.create_execution_ctx(),
    ).unwrap().into_array())]
    #[case::sliced(Patched::sparse(
        indices(&[1, 1030, 2000]),
        buffer![5i64, 6, 7].into_array(),
        3000,
        Scalar::from(1i64),
        &mut SESSION.create_execution_ctx(),
    ).unwrap().into_array().slice(2..1500).unwrap().optimize_ctx(&SESSION).unwrap())]
    #[case::floats(Patched::sparse(
        indices(&[1, 4]),
        PrimitiveArray::from_option_iter([Some(f64::NAN), None]).into_array(),
        6,
        Scalar::from(2.5f64),
        &mut SESSION.create_execution_ctx(),
    ).unwrap().into_array())]
    fn match_canonical(#[case] array: ArrayRef) {
        assert!(array.is::<Patched>());
        check(MinMax, NumericalAggregateOpts::default(), &array);
        check(Sum, NumericalAggregateOpts::default(), &array);
        check(NullCount, EmptyOptions, &array);
        check(IsConstant, EmptyOptions, &array);
        if array.dtype().is_float() {
            check(NanCount, EmptyOptions, &array);
        }
    }
}
