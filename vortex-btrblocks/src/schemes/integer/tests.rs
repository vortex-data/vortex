// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::LazyLock;

use itertools::Itertools;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Masked;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_compressor::CascadingCompressor;
use vortex_error::VortexResult;
use vortex_fastlanes::RLE;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
use crate::schemes::integer::IntRLEScheme;
static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_empty() -> VortexResult<()> {
    // Make sure empty array compression does not fail.
    let btr = BtrBlocksCompressor::default();
    let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable);
    let result = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_dict_encodable() -> VortexResult<()> {
    let mut codes = BufferMut::<i32>::with_capacity(65_535);
    // Write some runs of length 3 of a handful of different values. Interrupted by some
    // one-off values.

    let numbers = [0, 10, 50, 100, 1000, 3000]
        .into_iter()
        .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
        .collect_vec();

    let mut rng = StdRng::seed_from_u64(1u64);
    while codes.len() < 64000 {
        let run_length = rng.next_u32() % 5;
        let value = numbers[rng.next_u32() as usize % numbers.len()];
        for _ in 0..run_length {
            codes.push(value);
        }
    }

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(
        &codes.freeze().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Dict>());
    Ok(())
}

#[test]
fn constant_mostly_nulls() -> VortexResult<()> {
    let mut assertion_ctx = vortex_array::array_session().create_execution_ctx();
    let array = PrimitiveArray::new(
        buffer![189u8, 189, 189, 189, 189, 189, 189, 189, 189, 0, 46],
        Validity::from_iter(vec![
            false, false, false, false, false, false, false, false, false, false, true,
        ]),
    );
    let validity = array.validity()?;

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

    assert!(compressed.is::<Masked>());
    assert!(compressed.children()[0].is::<Constant>());

    let decoded = compressed;
    let expected =
        PrimitiveArray::new(buffer![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46], validity).into_array();
    assert_arrays_eq!(decoded, expected, &mut assertion_ctx);
    Ok(())
}

#[test]
fn nullable_sequence() -> VortexResult<()> {
    let mut assertion_ctx = vortex_array::array_session().create_execution_ctx();
    let values = (0i32..20).step_by(7).collect_vec();
    let array = PrimitiveArray::from_option_iter(values.clone().into_iter().map(Some));

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sequence>());

    let decoded = compressed;
    let expected = PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array();
    assert_arrays_eq!(decoded, expected, &mut assertion_ctx);
    Ok(())
}

#[test]
fn test_rle_compression() -> VortexResult<()> {
    let mut assertion_ctx = vortex_array::array_session().create_execution_ctx();
    let mut values = Vec::new();
    values.extend(iter::repeat_n(42i32, 100));
    values.extend(iter::repeat_n(123i32, 200));
    values.extend(iter::repeat_n(987i32, 150));

    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let compressor = CascadingCompressor::new(vec![&IntRLEScheme]);
    let compressed =
        compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<RLE>());

    let expected = Buffer::copy_from(&values).into_array();
    assert_arrays_eq!(compressed, expected, &mut assertion_ctx);
    Ok(())
}

#[test_with::env(CI)]
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
fn compress_large_int() -> VortexResult<()> {
    const NUM_LISTS: usize = 10_000;
    const ELEMENTS_PER_LIST: usize = 5_000;

    let prim = (0..NUM_LISTS)
        .flat_map(|list_idx| {
            (0..ELEMENTS_PER_LIST).map(move |elem_idx| (list_idx * 1000 + elem_idx) as f64)
        })
        .collect::<PrimitiveArray>()
        .into_array();

    let btr = BtrBlocksCompressor::default();
    btr.compress(&prim, &mut SESSION.create_execution_ctx())?;

    Ok(())
}
