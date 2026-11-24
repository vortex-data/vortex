// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::binary::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Affine;
use crate::AffineArray;
use crate::AffineArrayExt;
use crate::AffineArraySlotsExt;
use crate::AffineOptions;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

const MODES: [AffineOptions; 5] = [
    AffineOptions::FOR,
    AffineOptions::SCALE,
    AffineOptions::SLOPE,
    AffineOptions::ALL,
    AffineOptions::LECO,
];

const ALL_OPERATORS: [CompareOperator; 6] = [
    CompareOperator::Eq,
    CompareOperator::NotEq,
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

/// Timestamps in nanoseconds on a one-minute grid, with every seventh sample missing.
fn grid_timestamps() -> PrimitiveArray {
    PrimitiveArray::from_iter(
        (0..5000i64)
            .filter(|i| i % 7 != 3)
            .map(|i| 1_700_000_000_000_000_000 + i * 60_000_000_000),
    )
}

/// Millisecond timestamps in nanoseconds, roughly every 2.6 s with jitter.
fn jittered_timestamps() -> PrimitiveArray {
    PrimitiveArray::from_iter(
        (0..5000i64)
            .map(|i| 1_655_870_000_000_000_000 + i * 2_612_000_000 + (i * 7919 % 41) * 1_000_000),
    )
}

/// Prices on a 0.01 tick, stored in units of 1e-8.
fn prices() -> PrimitiveArray {
    PrimitiveArray::from_iter(
        (0..3000i64).map(|i| 6_776_562_000_000 + ((i * 37) % 501 - 250) * 1_000_000),
    )
}

fn nullable_ids() -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..4000u32).map(|i| (i % 11 != 0).then_some(3 * i + 17)))
}

fn extremes() -> PrimitiveArray {
    PrimitiveArray::from_iter([i64::MIN, i64::MAX, 0, -1, 1, i64::MIN + 1, i64::MAX - 1])
}

fn unsigned_extremes() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..2100u64).map(|i| u64::MAX - i * i * 1000))
}

fn small_signed() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..3000i32).map(|i| -128 + (i % 256)).map(|v| v as i8))
}

#[rstest]
#[case::grid(grid_timestamps())]
#[case::jitter(jittered_timestamps())]
#[case::prices(prices())]
#[case::nullable(nullable_ids())]
#[case::extremes(extremes())]
#[case::unsigned_extremes(unsigned_extremes())]
#[case::small_signed(small_signed())]
#[case::empty(PrimitiveArray::from_iter(Vec::<i64>::new()))]
#[case::all_null(PrimitiveArray::from_option_iter([None::<u16>, None, None]))]
fn roundtrip(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for options in MODES {
        let encoded = Affine::encode(&values, options, &mut ctx)?;
        assert_arrays_eq!(encoded, values, &mut ctx);
    }
    Ok(())
}

#[rstest]
#[case::start(0..100)]
#[case::within_chunk(1500..1600)]
#[case::across_chunks(1000..3500)]
#[case::to_end(2900..4285)]
fn slice(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = grid_timestamps();
    let expected = values.clone().into_array().slice(range.clone())?;
    for options in MODES {
        let encoded = Affine::encode(&values, options, &mut ctx)?.into_array();
        let sliced = encoded.slice(range.clone())?;
        assert_arrays_eq!(sliced, expected, &mut ctx);
        // Slicing twice exercises the offset of an already-sliced array.
        let inner = 7..range.len() / 2;
        assert_arrays_eq!(
            sliced.slice(inner.clone())?,
            expected.slice(inner)?,
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn scalar_at() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = nullable_ids();
    let expected = values.clone().into_array();
    let encoded = Affine::encode(&values, AffineOptions::ALL, &mut ctx)?.into_array();
    for index in [0, 1, 11, 1023, 1024, 2047, 3999] {
        assert_eq!(
            encoded.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}

#[test]
fn scale_finds_the_grid_step() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = Affine::encode(&prices(), AffineOptions::SCALE, &mut ctx)?;
    let scale = encoded
        .scales()
        .as_constant()
        .and_then(|s| s.as_primitive().typed_value::<i64>());
    assert_eq!(scale, Some(1_000_000));
    Ok(())
}

#[test]
fn slope_fits_a_regular_grid() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..4096i64).map(|i| 1_000 + 60 * i));
    let encoded = Affine::encode(&values, AffineOptions::SLOPE, &mut ctx)?;
    let residuals = encoded
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    // All-zero residuals are stored in the narrowest type.
    assert_eq!(residuals.ptype(), PType::U8);
    assert!(residuals.as_slice::<u8>().iter().all(|&r| r == 0));
    Ok(())
}

fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let buffers = array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(&buffer);
    }
    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

#[rstest]
#[case::grid(grid_timestamps())]
#[case::nullable(nullable_ids())]
fn serde(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for options in MODES {
        let encoded = Affine::encode(&values, options, &mut ctx)?.into_array();
        let read = serde_roundtrip(&encoded)?;
        assert_eq!(read.encoding_id(), encoded.encoding_id());
        assert_arrays_eq!(read, values, &mut ctx);
        let sliced = encoded.slice(1500..2600)?.optimize_ctx(&SESSION)?;
        assert_arrays_eq!(serde_roundtrip(&sliced)?, sliced, &mut ctx);
    }
    Ok(())
}

/// Bit-pack the residuals, narrow enough to need patches, so decoding takes the fused path.
#[rstest]
#[case::grid(grid_timestamps(), 3)]
#[case::grid_wide(grid_timestamps(), 40)]
#[case::jitter(jittered_timestamps(), 2)]
#[case::nullable(nullable_ids(), 1)]
fn fused_bitpacked(#[case] values: PrimitiveArray, #[case] bit_width: u8) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for options in MODES {
        let affine = Affine::encode(&values, options, &mut ctx)?;
        // BitPacked needs a width below the residual type's.
        let width = bit_width.min(affine.encoded().dtype().as_ptype().bit_width() as u8 - 1);
        let packed = BitPackedData::encode(affine.encoded(), width, &mut ctx)?.into_array();
        let fused = Affine::try_new(
            packed,
            affine.references().clone(),
            affine.scales().clone(),
            affine.slopes().clone(),
            0,
            affine.slope_shift(),
        )?
        .into_array();
        assert_arrays_eq!(fused, values, &mut ctx);
        for range in [1500..2600, 1024..2048, 7..900] {
            let expected = values.clone().into_array().slice(range.clone())?;
            assert_arrays_eq!(fused.slice(range)?, expected, &mut ctx);
        }
    }
    Ok(())
}

/// Dictionary-encoded residuals decode fused with the model, including slices and slopes.
#[rstest]
#[case::grid(grid_timestamps())]
#[case::jitter(jittered_timestamps())]
#[case::prices(prices())]
fn fused_dict(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for options in MODES {
        let affine = Affine::encode(&values, options, &mut ctx)?;
        let residuals = affine
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let dict = dict_encode(&residuals.into_array(), &mut ctx)?;
        let fused = Affine::try_new(
            dict.into_array(),
            affine.references().clone(),
            affine.scales().clone(),
            affine.slopes().clone(),
            0,
            affine.slope_shift(),
        )?
        .into_array();
        assert_arrays_eq!(fused, values, &mut ctx);
        for range in [1500..2600, 1024..2048, 7..900] {
            let expected = values.clone().into_array().slice(range.clone())?;
            assert_arrays_eq!(fused.slice(range)?, expected, &mut ctx);
        }
    }
    Ok(())
}

/// Bit-pack an Affine array's residuals at `bit_width` (clamped below residual type width).
fn bitpack_affine(
    affine: &AffineArray,
    bit_width: u8,
    ctx: &mut ExecutionCtx,
) -> VortexResult<AffineArray> {
    let max_width = affine.encoded().dtype().as_ptype().bit_width() as u8 - 1;
    let width = bit_width.min(max_width);
    let packed = BitPackedData::encode(affine.encoded(), width, ctx)?.into_array();
    Affine::try_new(
        packed,
        affine.references().clone(),
        affine.scales().clone(),
        affine.slopes().clone(),
        affine.offset(),
        affine.slope_shift(),
    )
}

#[rstest]
#[case::grid(grid_timestamps(), 3)]
#[case::jitter(jittered_timestamps(), 2)]
#[case::prices(prices(), 4)]
#[case::nullable(nullable_ids(), 1)]
#[case::extremes(unsigned_extremes(), 8)]
fn filter_bitpacked_and_primitive(
    #[case] values: PrimitiveArray,
    #[case] bit_width: u8,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(42);
    for options in MODES {
        let prim_affine = Affine::encode(&values, options, &mut ctx)?;
        let bp_affine = bitpack_affine(&prim_affine, bit_width, &mut ctx)?;
        for range in [
            0..values.len(),
            7..values.len().min(3500),
            1023..values.len().min(2050),
        ] {
            if range.is_empty() {
                continue;
            }
            let expected_slice = values.clone().into_array().slice(range.clone())?;
            let sliced_prim = prim_affine
                .clone()
                .into_array()
                .slice(range.clone())?
                .optimize_ctx(&SESSION)?;
            let sliced_bp = bp_affine
                .clone()
                .into_array()
                .slice(range.clone())?
                .optimize_ctx(&SESSION)?;
            let slice_len = range.len();

            for density in [0.005, 0.02, 0.15, 0.5, 0.99] {
                let bits = BitBuffer::from_iter((0..slice_len).map(|_| rng.random_bool(density)));
                for mask in [
                    Mask::from_buffer(bits.clone()),
                    Mask::from_indices(slice_len, bits.set_indices()),
                    Mask::from_slices(slice_len, bits.set_slices().collect()),
                ] {
                    let expected = expected_slice.filter(mask.clone())?;

                    let got_bp = <Affine as FilterKernel>::filter(
                        sliced_bp.as_::<Affine>(),
                        &mask,
                        &mut ctx,
                    )?
                    .expect("Affine filter kernel must engage on BitPacked child");
                    assert_arrays_eq!(got_bp, expected, &mut ctx);

                    let got_prim = <Affine as FilterKernel>::filter(
                        sliced_prim.as_::<Affine>(),
                        &mask,
                        &mut ctx,
                    )?
                    .expect("Affine filter kernel must engage on Primitive child");
                    assert_arrays_eq!(got_prim, expected, &mut ctx);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn filter_and_take_conformance_and_consistency() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in [
        PrimitiveArray::from_iter((0..2500i64).map(|i| 1_000_000 + (i % 73) * 1_000)),
        PrimitiveArray::from_option_iter(
            (0..2500i32).map(|i| (i % 9 != 0).then_some(-50_000 + i * 17)),
        ),
        PrimitiveArray::from_iter((0..1500u16).map(|i| 200 + (i % 64) * 5)),
    ] {
        for options in [
            AffineOptions::SCALE,
            AffineOptions::SLOPE,
            AffineOptions::ALL,
        ] {
            let affine = Affine::encode(&values, options, &mut ctx)?;
            let bp = bitpack_affine(&affine, 3, &mut ctx)?;
            test_filter_conformance(&bp.clone().into_array(), &mut ctx);
            test_take_conformance(&bp.clone().into_array(), &mut ctx);
            test_array_consistency(&bp.into_array(), &mut ctx);
        }
    }
    Ok(())
}

#[rstest]
#[case::grid(grid_timestamps(), 3)]
#[case::jitter(jittered_timestamps(), 2)]
#[case::prices(prices(), 4)]
#[case::nullable(nullable_ids(), 1)]
fn take_sparse_sliced_and_nullable_indices(
    #[case] values: PrimitiveArray,
    #[case] bit_width: u8,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(99);
    for options in MODES {
        let prim_affine = Affine::encode(&values, options, &mut ctx)?;
        let bp_affine = bitpack_affine(&prim_affine, bit_width, &mut ctx)?;
        for range in [
            0..values.len(),
            13..values.len().min(3200),
            1023..values.len().min(2050),
        ] {
            let expected_slice = values.clone().into_array().slice(range.clone())?;
            let sliced_prim = prim_affine
                .clone()
                .into_array()
                .slice(range.clone())?
                .optimize_ctx(&SESSION)?;
            let sliced_bp = bp_affine
                .clone()
                .into_array()
                .slice(range.clone())?
                .optimize_ctx(&SESSION)?;
            let slice_len = range.len();
            let k = (slice_len / 20).max(5);

            // Unsorted with duplicates (u32).
            let raw_indices: Vec<u32> = (0..k)
                .map(|_| rng.random_range(0..slice_len) as u32)
                .collect();
            let indices_u32 = PrimitiveArray::from_iter(raw_indices.clone()).into_array();
            let expected = expected_slice.take(indices_u32.clone())?;
            let got_bp =
                <Affine as TakeExecute>::take(sliced_bp.as_::<Affine>(), &indices_u32, &mut ctx)?
                    .expect("sparse take must engage on BitPacked child");
            assert_arrays_eq!(got_bp, expected, &mut ctx);
            let got_prim =
                <Affine as TakeExecute>::take(sliced_prim.as_::<Affine>(), &indices_u32, &mut ctx)?
                    .expect("sparse take must engage on Primitive child");
            assert_arrays_eq!(got_prim, expected, &mut ctx);

            // Nullable signed indices with out-of-bounds garbage at null slots.
            let mut vals = Vec::with_capacity(k);
            let mut valid_bits = Vec::with_capacity(k);
            for (pos, &idx) in raw_indices.iter().enumerate() {
                if pos % 5 == 0 {
                    vals.push(-999_999i64);
                    valid_bits.push(false);
                } else {
                    vals.push(i64::from(idx));
                    valid_bits.push(true);
                }
            }
            let nullable_idx =
                PrimitiveArray::new(Buffer::from_iter(vals), Validity::from_iter(valid_bits))
                    .into_array();
            let expected_null = expected_slice.take(nullable_idx.clone())?;
            let got_null_bp =
                <Affine as TakeExecute>::take(sliced_bp.as_::<Affine>(), &nullable_idx, &mut ctx)?
                    .expect("nullable sparse take must engage");
            assert_arrays_eq!(got_null_bp, expected_null, &mut ctx);
            let got_null_prim = <Affine as TakeExecute>::take(
                sliced_prim.as_::<Affine>(),
                &nullable_idx,
                &mut ctx,
            )?
            .expect("nullable sparse take must engage on Primitive child");
            assert_arrays_eq!(got_null_prim, expected_null, &mut ctx);

            // All-null indices.
            let all_null =
                PrimitiveArray::new(Buffer::from_iter(vec![999_999u32; 8]), Validity::AllInvalid)
                    .into_array();
            assert_arrays_eq!(
                sliced_bp.take(all_null.clone())?,
                expected_slice.take(all_null)?,
                &mut ctx
            );
        }
    }
    Ok(())
}

#[rstest]
#[case::grid(grid_timestamps(), 3)]
#[case::jitter(jittered_timestamps(), 2)]
#[case::prices(prices(), 4)]
#[case::nullable(nullable_ids(), 1)]
#[case::unsigned_extremes(unsigned_extremes(), 8)]
#[case::small_signed(small_signed(), 4)]
fn compare_matches_primitive_across_modes_and_slices(
    #[case] values: PrimitiveArray,
    #[case] bit_width: u8,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sample_positions = [0, 1, values.len() / 3, values.len() / 2, values.len() - 1];
    let mut constants: Vec<Scalar> = Vec::new();
    for pos in sample_positions {
        let s = values.execute_scalar(pos, &mut ctx)?;
        if !s.is_null() {
            constants.push(s);
        }
    }
    match values.ptype() {
        PType::I64 => {
            constants.extend([
                Scalar::from(i64::MIN),
                Scalar::from(i64::MAX),
                Scalar::from(0i64),
                Scalar::from(6_776_562_000_001i64),
                Scalar::from(1_700_000_000_000_000_001i64),
            ]);
        }
        PType::U32 => {
            constants.extend([
                Scalar::from(0u32),
                Scalar::from(18u32),
                Scalar::from(u32::MAX),
            ]);
        }
        PType::U64 => {
            constants.extend([Scalar::from(0u64), Scalar::from(u64::MAX)]);
        }
        PType::I8 => {
            constants.extend([
                Scalar::from(i8::MIN),
                Scalar::from(-1i8),
                Scalar::from(i8::MAX),
            ]);
        }
        _ => {}
    }

    for options in MODES {
        let prim_affine = Affine::encode(&values, options, &mut ctx)?;
        let bp_affine = bitpack_affine(&prim_affine, bit_width, &mut ctx).ok();
        for range in [
            0..values.len(),
            7..values.len().min(3200),
            1023..values.len().min(2050),
        ] {
            let expected_slice = values.clone().into_array().slice(range.clone())?;
            let sliced_prim = prim_affine
                .clone()
                .into_array()
                .slice(range.clone())?
                .optimize_ctx(&SESSION)?;
            let sliced_bp = bp_affine
                .as_ref()
                .map(|bp| {
                    bp.clone()
                        .into_array()
                        .slice(range.clone())?
                        .optimize_ctx(&SESSION)
                })
                .transpose()?;
            let slice_len = range.len();

            for scalar in &constants {
                for rhs_nullability in [Nullability::NonNullable, Nullability::Nullable] {
                    let rhs = ConstantArray::new(
                        scalar.cast(&scalar.dtype().with_nullability(rhs_nullability))?,
                        slice_len,
                    )
                    .into_array();
                    for op in ALL_OPERATORS {
                        let expected = expected_slice.binary(rhs.clone(), Operator::from(op))?;
                        if let Some(sliced_bp) = &sliced_bp {
                            let got_bp = <Affine as CompareKernel>::compare(
                                sliced_bp.as_::<Affine>(),
                                &rhs,
                                op,
                                &mut ctx,
                            )?
                            .expect("compare kernel must engage on BitPacked child");
                            assert_arrays_eq!(got_bp, expected, &mut ctx);
                        }

                        let got_prim = <Affine as CompareKernel>::compare(
                            sliced_prim.as_::<Affine>(),
                            &rhs,
                            op,
                            &mut ctx,
                        )?
                        .expect("compare kernel must engage on Primitive child");
                        assert_arrays_eq!(got_prim, expected, &mut ctx);
                    }
                }
            }

            let target = sliced_bp.as_ref().unwrap_or(&sliced_prim);
            if let Some(first_scalar) = constants.first() {
                let rhs = ConstantArray::new(first_scalar.clone(), slice_len).into_array();
                let actual_rev = rhs.binary(target.clone(), Operator::Lt)?;
                let expected_rev = rhs.binary(expected_slice.clone(), Operator::Lt)?;
                assert_arrays_eq!(actual_rev, expected_rev, &mut ctx);
            }
            let null_rhs =
                ConstantArray::new(Scalar::null(values.dtype().as_nullable()), slice_len)
                    .into_array();
            let actual_null = target.binary(null_rhs.clone(), Operator::Eq)?;
            let expected_null = expected_slice.binary(null_rhs, Operator::Eq)?;
            assert_arrays_eq!(actual_null, expected_null, &mut ctx);
        }
    }
    Ok(())
}

/// Negative scales, zero scales, wrapping arithmetic, and negative slopes constructed directly via
/// `Affine::try_new` must compare bit-identically to canonical execution across all 6 operators.
#[test]
fn compare_negative_scales_slopes_and_wrapping() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 3072usize;
    let residuals = PrimitiveArray::from_iter((0..len).map(|i| (i % 100) as u8));
    let packed = BitPackedData::encode(&residuals.clone().into_array(), 7, &mut ctx)?.into_array();

    // Chunk 0: negative scale (no wrap)
    // Chunk 1: zero scale (constant chunk)
    // Chunk 2: positive scale that wraps i16::MAX -> i16::MIN
    let references = PrimitiveArray::from_iter([500i16, -42, 32_000]).into_array();
    let scales = PrimitiveArray::from_iter([-5i16, 0, 20]).into_array();
    let slopes = ConstantArray::new(0i64, 3).into_array();

    for encoded in [packed, residuals.into_array()] {
        let affine = Affine::try_new(
            encoded,
            references.clone(),
            scales.clone(),
            slopes.clone(),
            0,
            0,
        )?
        .into_array();
        let canonical = affine
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();

        for rhs_val in [
            i16::MIN,
            -32_000,
            -43,
            -42,
            -41,
            0,
            5,
            6,
            250,
            251,
            499,
            500,
            501,
            32_000,
            32_760,
            i16::MAX,
        ] {
            let rhs = ConstantArray::new(rhs_val, len).into_array();
            for op in ALL_OPERATORS {
                let got =
                    <Affine as CompareKernel>::compare(affine.as_::<Affine>(), &rhs, op, &mut ctx)?
                        .expect("compare kernel must engage");
                let want = canonical.binary(rhs.clone(), Operator::from(op))?;
                assert_arrays_eq!(got, want, &mut ctx);
            }
        }
    }

    // Extreme slope that wraps `i64` in `slope_term` must fall back per chunk cleanly.
    let wrap_residuals = PrimitiveArray::from_iter((0..2048).map(|i| (i % 16) as u8));
    let wrap_packed =
        BitPackedData::encode(&wrap_residuals.into_array(), 4, &mut ctx)?.into_array();
    let wrap_affine = Affine::try_new(
        wrap_packed,
        PrimitiveArray::from_iter([0i64, i64::MAX - 10]).into_array(),
        PrimitiveArray::from_iter([3i64, -7]).into_array(),
        PrimitiveArray::from_iter([i64::MAX / 2, -(1i64 << 40)]).into_array(),
        0,
        4,
    )?
    .into_array();
    let wrap_canonical = wrap_affine
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
    for rhs_val in [i64::MIN, -100, 0, 100, i64::MAX - 10, i64::MAX] {
        let rhs = ConstantArray::new(rhs_val, 2048).into_array();
        for op in ALL_OPERATORS {
            let got = <Affine as CompareKernel>::compare(
                wrap_affine.as_::<Affine>(),
                &rhs,
                op,
                &mut ctx,
            )?
            .expect("compare kernel must engage");
            let want = wrap_canonical.binary(rhs.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
    }
    Ok(())
}

/// Non-divisible `Eq`/`NotEq` and out-of-range constants on unpatched non-nullable `SCALE`
/// arrays must short-circuit to a `ConstantArray`, while patches in short-circuited chunks must
/// still be evaluated accurately.
#[test]
fn compare_constant_short_circuit_and_patched_override() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let unpatched_vals = PrimitiveArray::from_iter((0..3072i64).map(|i| 1_000 + 10 * (i % 16)));
    let affine = Affine::encode(&unpatched_vals, AffineOptions::SCALE, &mut ctx)?;
    let bp = bitpack_affine(&affine, 4, &mut ctx)?;
    assert!(bp.encoded().as_::<BitPacked>().patches().is_none());

    let rhs_off_grid = ConstantArray::new(1_005i64, 3072).into_array();
    let eq_res = <Affine as CompareKernel>::compare(
        bp.as_view(),
        &rhs_off_grid,
        CompareOperator::Eq,
        &mut ctx,
    )?
    .expect("compare must engage");
    assert!(
        eq_res.as_constant().is_some(),
        "off-grid Eq should short-circuit to Constant"
    );
    assert_arrays_eq!(
        eq_res,
        ConstantArray::new(false, 3072).into_array(),
        &mut ctx
    );

    let rhs_high = ConstantArray::new(5_000i64, 3072).into_array();
    let lt_res =
        <Affine as CompareKernel>::compare(bp.as_view(), &rhs_high, CompareOperator::Lt, &mut ctx)?
            .expect("compare must engage");
    assert!(
        lt_res.as_constant().is_some(),
        "out-of-range Lt should short-circuit to Constant"
    );
    assert_arrays_eq!(
        lt_res,
        ConstantArray::new(true, 3072).into_array(),
        &mut ctx
    );

    // Patched override inside a chunk whose packed bounds [1000, 1150] are all < 5000.
    let patched_vals = PrimitiveArray::from_iter((0..3072i64).map(|i| {
        if i == 1500 {
            5_000
        } else {
            1_000 + 10 * (i % 16)
        }
    }));
    let affine_p = Affine::encode(&patched_vals, AffineOptions::SCALE, &mut ctx)?;
    let bp_p = bitpack_affine(&affine_p, 4, &mut ctx)?;
    assert!(bp_p.encoded().as_::<BitPacked>().patches().is_some());
    for op in ALL_OPERATORS {
        let got = <Affine as CompareKernel>::compare(bp_p.as_view(), &rhs_high, op, &mut ctx)?
            .expect("compare must engage");
        let want = patched_vals
            .clone()
            .into_array()
            .binary(rhs_high.clone(), Operator::from(op))?;
        assert_arrays_eq!(got, want, &mut ctx);
    }
    Ok(())
}

/// Bit-width 0 residuals (e.g. exact linear sequence) must support filter, take, and compare.
#[test]
fn zero_bit_width_filter_take_compare() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072i64).map(|i| 1_000 + 60 * i));
    let affine = Affine::encode(&values, AffineOptions::SLOPE, &mut ctx)?;
    let bp = bitpack_affine(&affine, 0, &mut ctx)?.into_array();

    let mask = Mask::from_indices(3072, [0, 5, 1023, 1024, 2500, 3071]);
    assert_arrays_eq!(bp.filter(mask.clone())?, values.filter(mask)?, &mut ctx);

    let indices = PrimitiveArray::from_iter([3071u32, 0, 1024, 5]).into_array();
    assert_arrays_eq!(bp.take(indices.clone())?, values.take(indices)?, &mut ctx);

    for rhs_val in [999i64, 1_000, 1_000 + 60 * 1024, 1_000 + 60 * 3071, 999_999] {
        let rhs = ConstantArray::new(rhs_val, 3072).into_array();
        for op in ALL_OPERATORS {
            assert_arrays_eq!(
                bp.binary(rhs.clone(), Operator::from(op))?,
                values
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(op))?,
                &mut ctx
            );
        }
    }
    Ok(())
}

#[test]
fn estimate_modes_ranks_candidates_accurately() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let price_est = Affine::estimate_modes(&prices(), 16, &mut ctx)?;
    assert!(
        price_est.for_bits - price_est.scale_bits >= 0.5,
        "prices should strongly prefer SCALE over FOR: {price_est:?}"
    );

    let grid_est = Affine::estimate_modes(&grid_timestamps(), 16, &mut ctx)?;
    assert!(
        grid_est.all_bits < grid_est.scale_bits && grid_est.for_bits - grid_est.all_bits >= 0.5,
        "grid timestamps should prefer ALL over SCALE/FOR: {grid_est:?}"
    );

    let mut rng = StdRng::seed_from_u64(123);
    let random = PrimitiveArray::from_iter((0..16_384u32).map(|_| rng.random_range(0..10_000)));
    let rand_est = Affine::estimate_modes(&random, 16, &mut ctx)?;
    assert!(
        rand_est.for_bits - rand_est.scale_bits < 0.5
            && rand_est.for_bits - rand_est.slope_bits < 0.5
            && rand_est.for_bits - rand_est.all_bits < 0.5,
        "random data should not beat FOR by >= 0.5 bits/val: {rand_est:?}"
    );
    Ok(())
}

/// Filter, take, and compare must work identically when `encoded` is a `Dict` child or when
/// `references`/`scales`/`slopes` and `BitPacked` patch values are themselves compressed.
#[test]
fn compute_kernels_with_dict_child_and_compressed_patches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..4096i64).map(|i| {
        if i % 37 == 0 {
            1_000_000 + 100 * (200 + (i % 50))
        } else {
            1_000_000 + (i / 1024) * 50_000 + 100 * (i % 15)
        }
    }));
    let affine = Affine::encode(&values, AffineOptions::SCALE, &mut ctx)?;

    // 1. Dict child with compressed references.
    let residuals = affine
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let dict_encoded = dict_encode(&residuals.into_array(), &mut ctx)?.into_array();
    let dict_refs = dict_encode(affine.references(), &mut ctx)?.into_array();
    let dict_affine = Affine::try_new(
        dict_encoded,
        dict_refs,
        affine.scales().clone(),
        affine.slopes().clone(),
        0,
        affine.slope_shift(),
    )?
    .into_array()
    .slice(5..3900)?
    .optimize_ctx(&SESSION)?;
    let expected_slice = values.clone().into_array().slice(5..3900)?;

    let sparse_mask = Mask::from_indices(3895, [0, 32, 1020, 1021, 2500, 3894]);
    assert_arrays_eq!(
        dict_affine.filter(sparse_mask.clone())?,
        expected_slice.filter(sparse_mask)?,
        &mut ctx
    );
    let take_idx = PrimitiveArray::from_iter([3894u32, 0, 1021, 32, 32]).into_array();
    assert_arrays_eq!(
        dict_affine.take(take_idx.clone())?,
        expected_slice.take(take_idx)?,
        &mut ctx
    );
    let rhs = ConstantArray::new(1_050_500i64, 3895).into_array();
    for op in ALL_OPERATORS {
        assert_arrays_eq!(
            dict_affine.binary(rhs.clone(), Operator::from(op))?,
            expected_slice.binary(rhs.clone(), Operator::from(op))?,
            &mut ctx
        );
    }

    // 2. BitPacked child whose patch values are themselves BitPacked (exercising both sparse and
    // dense patch-value paths in `apply_filtered_patches`).
    let bp = BitPackedData::encode(affine.encoded(), 4, &mut ctx)?;
    let orig_patches = bp.patches().expect("outliers must create patches");
    let compressed_patch_vals =
        BitPackedData::encode(orig_patches.values(), 9, &mut ctx)?.into_array();
    let compressed_patches = orig_patches.map_values(|_| Ok(compressed_patch_vals))?;
    let bp_parts = BitPacked::into_parts(bp);
    let bp_with_compressed_patches = BitPacked::try_new(
        bp_parts.packed,
        affine.encoded().dtype().as_ptype(),
        bp_parts.validity,
        Some(compressed_patches),
        4,
        bp_parts.len,
        bp_parts.offset,
    )?
    .into_array();
    let bp_affine = Affine::try_new(
        bp_with_compressed_patches,
        affine.references().clone(),
        affine.scales().clone(),
        affine.slopes().clone(),
        0,
        affine.slope_shift(),
    )?
    .into_array();

    // Very sparse filter (hits < patches / 8) and dense filter (hits >= patches / 8).
    let very_sparse = Mask::from_indices(4096, [0, 37, 74, 1500]);
    assert_arrays_eq!(
        bp_affine.filter(very_sparse.clone())?,
        values.filter(very_sparse)?,
        &mut ctx
    );
    let dense = Mask::from_iter((0..4096).map(|i| i % 2 == 0));
    assert_arrays_eq!(
        bp_affine.filter(dense.clone())?,
        values.filter(dense)?,
        &mut ctx
    );
    Ok(())
}

/// Direct kernel calls must handle empty sliced arrays (`len == 0, offset > 0`), null constants,
/// constant-null patches, and empty / all-null take indices without panicking.
#[test]
fn compute_kernels_edge_cases() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let base = bitpack_affine(
        &Affine::encode(&prices(), AffineOptions::SCALE, &mut ctx)?,
        10,
        &mut ctx,
    )?;
    // Empty slice with non-zero offset.
    let empty_sliced = Affine::try_new(
        base.encoded().slice(5..5)?.optimize_ctx(&SESSION)?,
        base.references().slice(0..1)?.optimize_ctx(&SESSION)?,
        base.scales().slice(0..1)?.optimize_ctx(&SESSION)?,
        base.slopes().slice(0..1)?.optimize_ctx(&SESSION)?,
        5,
        base.slope_shift(),
    )?;
    let empty_rhs = ConstantArray::new(0i64, 0).into_array();
    let cmp_empty = <Affine as CompareKernel>::compare(
        empty_sliced.as_view(),
        &empty_rhs,
        CompareOperator::Lt,
        &mut ctx,
    )?
    .expect("compare should handle empty sliced Affine");
    assert_eq!(cmp_empty.len(), 0);
    assert_eq!(cmp_empty.dtype(), &DType::Bool(Nullability::NonNullable));

    // Direct compare against a null constant.
    let null_rhs = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        base.len(),
    )
    .into_array();
    let cmp_null = <Affine as CompareKernel>::compare(
        base.as_view(),
        &null_rhs,
        CompareOperator::Eq,
        &mut ctx,
    )?
    .expect("compare should handle null constant RHS");
    assert_eq!(cmp_null.len(), base.len());
    assert!(cmp_null.all_invalid(&mut ctx)?);

    // Empty and all-null take indices (both BitPacked and non-BitPacked children).
    let empty_idx = PrimitiveArray::from_iter(std::iter::empty::<u32>()).into_array();
    let all_null_idx = PrimitiveArray::new(
        Buffer::from_iter([0u32, 1, 2]),
        Validity::Array(BoolArray::from(BitBuffer::new_unset(3)).into_array()),
    )
    .into_array();
    let prim_affine = Affine::encode(&prices(), AffineOptions::SCALE, &mut ctx)?;
    for aff in [base.as_view(), prim_affine.as_view()] {
        let taken_empty = <Affine as TakeExecute>::take(aff, &empty_idx, &mut ctx)?
            .expect("empty take should be handled");
        assert_eq!(taken_empty.len(), 0);
        let taken_null = <Affine as TakeExecute>::take(aff, &all_null_idx, &mut ctx)?
            .expect("all-null take should be handled");
        assert_eq!(taken_null.len(), 3);
        assert!(taken_null.all_invalid(&mut ctx)?);
    }

    // Nullable BitPacked child whose patch values are `ConstantArray(null)`.
    let len = 2048usize;
    let validity = Validity::from_iter((0..len).map(|i| i != 17 && i != 1500));
    let packed_zeros = ByteBuffer::zeroed(2 * 128 * 4);
    let null_patches = Patches::new(
        len,
        0,
        PrimitiveArray::from_iter([17u32, 1500]).into_array(),
        ConstantArray::new(
            Scalar::null(DType::Primitive(PType::U16, Nullability::Nullable)),
            2,
        )
        .into_array(),
        None,
        &mut ctx,
    )?;
    let bp_null_patches = BitPacked::try_new(
        BufferHandle::new_host(packed_zeros),
        PType::U16,
        validity,
        Some(null_patches),
        4,
        len,
        0,
    )?
    .into_array();
    let aff_null_patches = Affine::try_new(
        bp_null_patches,
        ConstantArray::new(100i64, 2).into_array(),
        ConstantArray::new(5i64, 2).into_array(),
        ConstantArray::new(0i64, 2).into_array(),
        0,
        16,
    )?;
    let mask = Mask::from_indices(len, [0, 17, 500, 1500, 2047]);
    let filtered = <Affine as FilterKernel>::filter(aff_null_patches.as_view(), &mask, &mut ctx)?
        .expect("filter should handle constant-null patches");
    let expected = PrimitiveArray::new(
        Buffer::from_iter([100i64; 5]),
        Validity::from_iter([true, false, true, false, true]),
    );
    assert_arrays_eq!(filtered, expected, &mut ctx);
    Ok(())
}
