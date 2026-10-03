// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Affine;
use crate::AffineArrayExt;
use crate::BitPackedData;
use crate::AffineArraySlotsExt;
use crate::AffineOptions;
use crate::VarBitPacked;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
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
        (0..5000i64).map(|i| 1_655_870_000_000_000_000 + i * 2_612_000_000 + (i * 7919 % 41) * 1_000_000),
    )
}

/// Prices on a 0.01 tick, stored in units of 1e-8.
fn prices() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..3000i64).map(|i| 6_776_562_000_000 + ((i * 37) % 501 - 250) * 1_000_000))
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
        assert_arrays_eq!(sliced.slice(inner.clone())?, expected.slice(inner)?, &mut ctx);
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
    let residuals = encoded.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
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
        let sliced = encoded.slice(1500..2600)?;
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

/// Residuals packed at one width per chunk decode fused with the model, including slices.
#[rstest]
#[case::grid(grid_timestamps())]
#[case::jitter(jittered_timestamps())]
#[case::prices(prices())]
#[case::nullable(nullable_ids())]
#[case::unsigned_extremes(unsigned_extremes())]
fn fused_var_bitpacked(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for options in MODES {
        let affine = Affine::encode(&values, options, &mut ctx)?;
        let residuals = affine.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let packed = VarBitPacked::encode(&residuals, &mut ctx)?.into_array();
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
        for range in [1500..2000, 1024..2048, 7..900] {
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
        let residuals = affine.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let dict = vortex_array::builders::dict::dict_encode(&residuals.into_array(), &mut ctx)?;
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
