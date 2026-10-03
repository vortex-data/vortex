// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Test data is built with intentional truncating casts.
#![allow(clippy::cast_possible_truncation)]

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::EntropyBins;

/// Deterministic xorshift values with a skewed, clustered distribution.
fn skewed(len: usize, seed: u64) -> Vec<i64> {
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

fn roundtrip<T: NativePType>(values: Vec<T>) -> VortexResult<()> {
    roundtrip_with(values.clone(), false)?;
    roundtrip_with(values, true)
}

fn roundtrip_with<T: NativePType>(values: Vec<T>, delta: bool) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
    let encoded = EntropyBins::from_primitive(array.as_view(), 8, delta)?;
    assert_arrays_eq!(
        encoded.clone().into_array(),
        array.clone().into_array(),
        &mut ctx
    );
    // Point lookups, including block edges.
    let n = array.len();
    for i in [0, 1, 15, 16, 1023, 1024, 1025, n / 2, n - 1] {
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
    roundtrip(vec![42u32; 3000])
}

#[test]
fn nullable_and_slices() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = skewed(5000, 3);
    let validity = Validity::from_iter((0..5000).map(|i| i % 7 != 0));
    let array = PrimitiveArray::new(Buffer::from(values), validity);
    let encoded = EntropyBins::from_primitive(array.as_view(), 8, true)?.into_array();
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
        for (array, delta) in [
            (wide.clone(), false),
            (wide, true),
            (narrow.clone(), false),
            (narrow, true),
        ] {
            let encoded = EntropyBins::from_primitive(array.as_view(), 8, delta)?.into_array();
            crate::x86::set_force_scalar(true);
            let scalar = encoded.clone().execute::<PrimitiveArray>(&mut ctx)?;
            crate::x86::set_force_scalar(false);
            let simd = encoded.execute::<PrimitiveArray>(&mut ctx)?;
            assert_arrays_eq!(scalar.into_array(), simd.into_array(), &mut ctx);
        }
    }
    Ok(())
}
