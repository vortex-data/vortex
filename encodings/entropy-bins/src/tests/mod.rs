// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Test data is built with intentional truncating casts.
#![allow(clippy::cast_possible_truncation)]

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::NativePType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

use crate::BLOCK_VALUES;
use crate::EntropyBins;
use crate::EntropyBinsConfig;
use crate::EntropyBinsOptions;
use crate::MAX_BLOCK_VALUES;

/// Deterministic xorshift values with a skewed, clustered distribution.
pub(crate) fn skewed(len: usize, seed: u64) -> Vec<i64> {
    let mut state = seed | 1;
    let mut value = 0i64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let jump = (state >> 20) as i64;
            value += match state % 100 {
                0..60 => 0,
                60..90 => jump % 9 - 4,
                90..99 => jump % 513 - 256,
                _ => jump % 1_000_001 - 500_000,
            };
            value
        })
        .collect()
}

mod conformance;
mod malformed;
mod pushdown;

const LAGS: [usize; 6] = [0, 1, 2, 3, 4, 8];

const BLOCK_SIZES: [usize; 3] = [BLOCK_VALUES, 2 * BLOCK_VALUES, MAX_BLOCK_VALUES];

fn roundtrip<T: NativePType>(values: Vec<T>) -> VortexResult<()> {
    for lag in LAGS {
        roundtrip_with(values.clone(), EntropyBinsOptions::new(lag, BLOCK_VALUES))?;
    }
    for block_values in [2 * BLOCK_VALUES, MAX_BLOCK_VALUES] {
        roundtrip_with(values.clone(), EntropyBinsOptions::new(0, block_values))?;
        roundtrip_with(values.clone(), EntropyBinsOptions::new(3, block_values))?;
    }
    for (lag, block_values) in [(0, BLOCK_VALUES), (1, MAX_BLOCK_VALUES)] {
        let options = EntropyBinsOptions::new(lag, block_values).with_word_bits(8);
        roundtrip_with(values.clone(), options)?;
    }
    Ok(())
}

fn roundtrip_with<T: NativePType>(values: Vec<T>, options: EntropyBinsOptions) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
    let encoded = EntropyBins::from_primitive(array.as_view(), 8, options)?;
    assert_arrays_eq!(
        encoded.clone().into_array(),
        array.clone().into_array(),
        &mut ctx
    );
    // Point lookups, including block edges.
    let n = array.len();
    for i in [
        0,
        1,
        2,
        7,
        8,
        15,
        16,
        1023,
        1024,
        1025,
        1031,
        4095,
        4096,
        4097,
        n / 2,
        n - 1,
    ] {
        if i < n {
            assert_eq!(
                encoded.clone().into_array().execute_scalar(i, &mut ctx)?,
                array.clone().into_array().execute_scalar(i, &mut ctx)?
            );
        }
    }
    Ok(())
}

#[rstest]
#[case(1)]
#[case(17)]
#[case(1024)]
#[case(1025)]
#[case(5000)]
#[case(300_000)]
fn roundtrip_i64(#[case] n: usize) -> VortexResult<()> {
    roundtrip(skewed(n, 7))
}

#[rstest]
#[case(1)]
#[case(4097)]
#[case(70_000)]
fn roundtrip_narrow(#[case] n: usize) -> VortexResult<()> {
    let v = skewed(n, 11);
    roundtrip(v.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
    roundtrip(v.iter().map(|&x| x as i16).collect::<Vec<_>>())?;
    roundtrip(v.iter().map(|&x| x as i8).collect::<Vec<_>>())?;
    roundtrip(
        v.iter()
            .map(|&x| x.unsigned_abs() as u32)
            .collect::<Vec<_>>(),
    )?;
    roundtrip(
        v.iter()
            .map(|&x| x.unsigned_abs() as u16)
            .collect::<Vec<_>>(),
    )?;
    roundtrip(
        v.iter()
            .map(|&x| x.unsigned_abs() as u8)
            .collect::<Vec<_>>(),
    )?;
    roundtrip(v.iter().map(|&x| x.unsigned_abs()).collect::<Vec<_>>())
}

#[test]
fn roundtrip_extremes() -> VortexResult<()> {
    roundtrip(vec![
        i64::MIN,
        i64::MAX,
        0,
        -1,
        1,
        i64::MIN + 1,
        i64::MAX - 1,
    ])?;
    roundtrip(vec![u64::MAX, 0, 1, u64::MAX - 1])?;
    // Differences of exactly 2^63.
    roundtrip(
        (0..3000)
            .map(|i| if i % 2 == 0 { i64::MIN } else { 0 })
            .collect(),
    )?;
    roundtrip(vec![42u32; 3000])
}

/// Rare outliers give a chunk a bin too wide for the vector merges; blocks without them still
/// take the vector path.
#[rstest]
#[case(0)]
#[case(1)]
fn wide_outlier_bins(#[case] lag: usize) -> VortexResult<()> {
    let small = skewed(20_000, 5);
    let mut values: Vec<i64> = small.iter().map(|v| v % 1000).collect();
    values[5000] = i64::MIN;
    values[5001] = i64::MAX;
    roundtrip_with(values.clone(), EntropyBinsOptions::new(lag, BLOCK_VALUES))?;
    let narrow: Vec<i32> = values
        .iter()
        .map(|&v| match v {
            i64::MIN => i32::MIN,
            i64::MAX => i32::MAX,
            v => v as i32,
        })
        .collect();
    roundtrip_with(narrow, EntropyBinsOptions::new(lag, BLOCK_VALUES))
}

fn serde_roundtrip_with<T: NativePType>(values: Vec<T>, nullable: bool) -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let validity = if nullable {
        Validity::from_iter((0..values.len()).map(|i| i % 5 != 2))
    } else {
        Validity::NonNullable
    };
    let array = PrimitiveArray::new(Buffer::from(values), validity);
    for lag in [0, 1, 3] {
        let encoded = EntropyBins::from_primitive(
            array.as_view(),
            8,
            EntropyBinsOptions::new(lag, BLOCK_VALUES),
        )?
        .into_array();
        let n = encoded.len();
        // A slice keeps the unsliced buffers and validity, so it serializes its start.
        for range in [0..n, 37.min(n)..n.saturating_sub(11).max(37.min(n))] {
            let sliced = encoded.slice(range.clone())?;
            let ctx = ArrayContext::empty();
            let mut bytes = ByteBufferMut::empty();
            for buffer in sliced.serialize(&ctx, &session, &SerializeOptions::default())? {
                bytes.extend_from_slice(buffer.as_ref());
            }
            let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
                sliced.dtype(),
                sliced.len(),
                &ReadContext::new(ctx.to_ids()),
                &session,
            )?;
            assert_arrays_eq!(
                decoded,
                array.clone().into_array().slice(range)?,
                &mut session.create_execution_ctx()
            );
        }
    }
    Ok(())
}

#[test]
fn serde_roundtrip() -> VortexResult<()> {
    let values = skewed(300_000, 3);
    serde_roundtrip_with(values.clone(), false)?;
    serde_roundtrip_with(values.iter().map(|&v| v as u64).collect::<Vec<_>>(), true)?;
    serde_roundtrip_with(values.iter().map(|&v| v as i16).collect::<Vec<_>>(), true)?;
    serde_roundtrip_with(vec![u8::MAX; 10], false)?;
    serde_roundtrip_with(Vec::<i32>::new(), false)
}

/// Interleaved series (`x`, `y`, `z` per row group) are smallest with the matching lag.
#[test]
fn interleaved_series_prefers_lag() -> VortexResult<()> {
    let series: Vec<Vec<i64>> = (0..3).map(|c| skewed(50_000, 9 + c)).collect();
    let values: Vec<i64> = (0..3 * 50_000)
        .map(|i| series[i % 3][i / 3] + 1_000_000_000 * (i % 3) as i64)
        .collect();
    let array = PrimitiveArray::new(Buffer::from(values.clone()), Validity::NonNullable);
    let size = |lag| -> VortexResult<usize> {
        let e = EntropyBins::from_primitive(
            array.as_view(),
            8,
            EntropyBinsOptions::new(lag, BLOCK_VALUES),
        )?;
        Ok(e.data().data.len())
    };
    assert!(size(3)? < size(1)?);
    assert_eq!(
        EntropyBins::plan(array.as_view(), &EntropyBinsConfig::BALANCED)?
            .options
            .lag,
        3
    );
    roundtrip(values)
}

#[test]
fn nullable_and_slices() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = skewed(5000, 3);
    let validity = Validity::from_iter((0..5000).map(|i| i % 7 != 0));
    let array = PrimitiveArray::new(Buffer::from(values), validity);
    let encoded =
        EntropyBins::from_primitive(array.as_view(), 8, EntropyBinsOptions::new(1, BLOCK_VALUES))?
            .into_array();
    assert_arrays_eq!(encoded, array.clone().into_array(), &mut ctx);
    for (a, b) in [(0, 1), (3, 1500), (1024, 2048), (1000, 5000), (4999, 5000)] {
        assert_arrays_eq!(
            encoded.slice(a..b)?,
            array.clone().into_array().slice(a..b)?,
            &mut ctx
        );
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[test]
fn simd_matches_scalar() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for (seed, n) in [(1u64, 70_000usize), (2, 1500), (3, 300_000)] {
        let v = skewed(n, seed);
        let wide = PrimitiveArray::new(Buffer::from(v.clone()), Validity::NonNullable);
        let narrow = PrimitiveArray::new(
            Buffer::from(v.iter().map(|&x| x as i16).collect::<Vec<_>>()),
            Validity::NonNullable,
        );
        let cases = LAGS.iter().flat_map(|&lag| {
            BLOCK_SIZES.iter().flat_map(move |&bv| {
                [8, 16].into_iter().flat_map(move |wb| {
                    let options = EntropyBinsOptions::new(lag, bv).with_word_bits(wb);
                    [(false, options), (true, options)]
                })
            })
        });
        for (is_narrow, options) in cases {
            let array = if is_narrow { &narrow } else { &wide };
            let encoded = EntropyBins::from_primitive(array.as_view(), 8, options)?.into_array();
            crate::x86::set_force_scalar(true);
            let scalar = encoded.clone().execute::<PrimitiveArray>(&mut ctx)?;
            crate::x86::set_force_scalar(false);
            let simd = encoded.execute::<PrimitiveArray>(&mut ctx)?;
            assert_arrays_eq!(scalar.into_array(), simd.into_array(), &mut ctx);
        }
    }
    Ok(())
}

/// Repeated probes cache a block after its second touch; every access pattern must still read
/// the right rows, including nulls and slices.
#[rstest]
#[case(0, BLOCK_VALUES, 16)]
#[case(1, BLOCK_VALUES, 16)]
#[case(3, BLOCK_VALUES, 16)]
#[case(1, MAX_BLOCK_VALUES, 16)]
#[case(0, BLOCK_VALUES, 8)]
#[case(1, MAX_BLOCK_VALUES, 8)]
fn repeated_probe(
    #[case] lag: usize,
    #[case] block_values: usize,
    #[case] word_bits: u32,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let n = 5000;
    let validity = Validity::from_iter((0..n).map(|i| i % 11 != 0));
    let parray = PrimitiveArray::new(Buffer::from(skewed(n, 13)), validity);
    let options = EntropyBinsOptions::new(lag, block_values).with_word_bits(word_bits);
    let encoded = EntropyBins::from_primitive(parray.as_view(), 8, options)?.into_array();
    let array = parray.into_array();
    for (a, b) in [(0, n), (700, 4321)] {
        let expected = array.slice(a..b)?;
        let sliced = encoded.slice(a..b)?;
        let mut probe = sliced.repeated_probe();
        // Revisit blocks, jump between them and walk one block backwards.
        let len = b - a;
        let order = (0..len)
            .step_by(97)
            .chain([5, 6, 5, len - 1, 0, 1030, 1029, 1028, 4100, 4099])
            .chain((0..len.min(block_values)).rev());
        for i in order.filter(|&i| i < len) {
            assert_eq!(
                probe.execute_scalar(i, &mut ctx)?,
                expected.execute_scalar(i, &mut ctx)?,
                "row {i} of {a}..{b}"
            );
        }
    }
    Ok(())
}

/// Each preset stays within its dials, and the smaller presets are not larger.
#[test]
fn presets_respect_their_dials() -> VortexResult<()> {
    let v: Vec<i16> = skewed(200_000, 17)
        .iter()
        .map(|&x| (x % 50) as i16)
        .collect();
    let array = PrimitiveArray::new(Buffer::from(v), Validity::NonNullable);
    let size = |config: &EntropyBinsConfig| -> VortexResult<(EntropyBinsOptions, usize)> {
        let plan = EntropyBins::plan(array.as_view(), config)?;
        let encoded = EntropyBins::from_primitive(array.as_view(), config.level, plan.options)?;
        Ok((plan.options, encoded.data().data.len()))
    };
    let (fast, fast_bytes) = size(&EntropyBinsConfig::FAST)?;
    assert_eq!(fast.block_values, BLOCK_VALUES);
    assert_eq!(fast.word_bits, 16);
    assert!(fast.lag <= 1);
    let (_, balanced_bytes) = size(&EntropyBinsConfig::BALANCED)?;
    let (_, smallest_bytes) = size(&EntropyBinsConfig::SMALLEST)?;
    assert!(smallest_bytes <= balanced_bytes && balanced_bytes <= fast_bytes);
    Ok(())
}
