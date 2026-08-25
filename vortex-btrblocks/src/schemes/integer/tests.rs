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
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_compressor::CascadingCompressor;
use vortex_error::VortexExpect;
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
    let mut ctx = SESSION.create_execution_ctx();
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
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn nullable_sequence() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0i32..20).step_by(7).collect_vec();
    let array = PrimitiveArray::from_option_iter(values.clone().into_iter().map(Some));

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sequence>());

    let decoded = compressed;
    let expected = PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array();
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_rle_compression() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
    assert_arrays_eq!(compressed, expected, &mut ctx);
    Ok(())
}

/// Compresses 50M values, so it is ignored by default and run only by the "Rust tests
/// (linux-musl)" CI job. Setting `VORTEX_SKIP_SLOW_TESTS` at build time drops it from the
/// binary, which is how the sanitizer jobs avoid compiling it at all. To run it locally:
///
/// ```text
/// cargo test --release -p vortex-btrblocks compress_large_int -- --ignored
/// ```
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
#[test]
#[ignore = "slow: compresses 50M values, run by the \"Rust tests (linux-musl)\" CI job"]
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

/// The compressor picks ALP exponents from a sample, so values the sample did not represent must
/// be stored as patches, indexed per chunk. This is the structure `compress_large_int` reaches
/// only by scale; here the misfit values are placed deliberately, which also pins the
/// `patch_chunk_offsets` width across the three magnitudes it is chosen from.
#[rstest::rstest]
#[case::sparse_patches(200_000, 1_000, PType::U8)]
#[case::dense_patches(200_000, 100, PType::U16)]
#[case::many_patches(1_000_000, 10, PType::U32)]
fn alp_patches_are_chunk_indexed(
    #[case] len: usize,
    #[case] patch_every: usize,
    #[case] chunk_offsets_ptype: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // Whole numbers dominate, so the sampled exponents encode them exactly; the sprinkled values
    // need more decimal digits than those exponents can represent and must be patched.
    let values = (0..len)
        .map(|i| {
            if i % patch_every == patch_every - 1 {
                i as f64 + 0.123_456_789_012_345
            } else {
                i as f64
            }
        })
        .collect::<PrimitiveArray>()
        .into_array();

    let compressed = BtrBlocksCompressor::default().compress(&values, &mut ctx)?;

    let offsets = compressed
        .children_names()
        .iter()
        .position(|name| name == "patch_chunk_offsets")
        .map(|idx| compressed.children()[idx].clone())
        .vortex_expect("compressed array must carry chunk-indexed ALP patches");
    assert_eq!(offsets.dtype().as_ptype(), chunk_offsets_ptype);

    assert_arrays_eq!(compressed, values, &mut ctx);
    Ok(())
}
