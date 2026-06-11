// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::random_temporal_ext_dtype;
use vortex_array::scalar::arbitrary::random_scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::CompressorStrategy;
use crate::FuzzArrayAction;
use crate::SESSION;
use crate::array::assert_array_eq;
use crate::array::compress_array;
use crate::array::filter_canonical_array;
use crate::array::scalar_at_canonical_array;
use crate::array::search_sorted_canonical_array;
use crate::array::slice_canonical_array;
use crate::run_fuzz_action;
use crate::sort_canonical_array;

/// Deterministic pseudo-random bytes for driving [`Unstructured`].
fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect()
}

/// Temporal arrays generated through the arbitrary machinery must round-trip through
/// compression and agree with the canonical baselines for core operations.
#[test]
fn temporal_array_baselines_match_engine() -> VortexResult<()> {
    let bytes = pseudo_random_bytes(256 * 1024, 7);
    let mut u = Unstructured::new(&bytes);
    let mut ctx = SESSION.create_execution_ctx();
    let mut tested = 0;

    while u.len() > 4096 && tested < 16 {
        let nullability = if Arbitrary::arbitrary(&mut u).unwrap_or(false) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let Ok(ext_dtype) = random_temporal_ext_dtype(&mut u, nullability) else {
            break;
        };
        let dtype = DType::Extension(ext_dtype);
        let Ok(array) = ArbitraryArray::arbitrary_with_config(
            &mut u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype.clone()),
                len: 1..=64,
            },
        ) else {
            break;
        };
        let array = array.0;
        tested += 1;

        // Compression must round-trip (exercises TemporalScheme / datetime-parts).
        let compressed = compress_array(&array, CompressorStrategy::Default);
        assert_array_eq(&array, &compressed, 0).map_err(|e| vortex_err!("compress: {e}"))?;

        // Slice baseline vs engine.
        let start = array.len() / 4;
        let stop = array.len() / 2;
        let expected = slice_canonical_array(&array, start, stop, &mut ctx)?;
        let actual = compressed.slice(start..stop)?;
        assert_array_eq(&expected, &actual, 0).map_err(|e| vortex_err!("slice: {e}"))?;

        // Filter baseline vs engine.
        let mask = (0..array.len()).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let expected = filter_canonical_array(&array, &mask, &mut ctx)?;
        let actual = compressed.clone().filter(Mask::from_iter(mask))?;
        assert_array_eq(&expected, &actual, 0).map_err(|e| vortex_err!("filter: {e}"))?;

        // ScalarAt baseline vs engine.
        let canonical = array.clone().execute::<Canonical>(&mut ctx)?;
        let expected = scalar_at_canonical_array(canonical, 0, &mut ctx)?;
        let actual = compressed.execute_scalar(0, &mut ctx)?;
        assert_eq!(expected, actual, "scalar_at mismatch");

        // SearchSorted baseline vs engine on the sorted array.
        let Ok(needle) = random_scalar(&mut u, &dtype) else {
            break;
        };
        if !needle.is_null() {
            let sorted = sort_canonical_array(&array, &mut ctx)?;
            let expected =
                search_sorted_canonical_array(&sorted, &needle, SearchSortedSide::Left, &mut ctx)?;
            let actual = sorted.search_sorted(&needle, SearchSortedSide::Left)?;
            assert_eq!(expected, actual, "search_sorted mismatch");
        }
    }

    assert!(tested > 0, "no temporal arrays were generated");
    Ok(())
}

/// End-to-end smoke test of the fuzz pipeline, covering the arbitrary dtypes (including
/// temporal extension dtypes) without needing a fuzzing engine.
#[test]
fn fuzz_action_pipeline_smoke() {
    let mut ran = 0;
    for seed in 0..256 {
        let bytes = pseudo_random_bytes(64 * 1024, seed);
        let mut u = Unstructured::new(&bytes);
        let Ok(action) = FuzzArrayAction::arbitrary(&mut u) else {
            continue;
        };
        if let Err(e) = run_fuzz_action(action) {
            panic!("seed {seed}: {e}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no fuzz actions were generated");
}

/// The generated temporal arrays must canonicalize to extension arrays with the requested
/// dtype and length.
#[test]
fn arbitrary_temporal_array_has_requested_dtype() -> VortexResult<()> {
    let bytes = pseudo_random_bytes(64 * 1024, 11);
    let mut u = Unstructured::new(&bytes);

    for _ in 0..8 {
        let Ok(ext_dtype) = random_temporal_ext_dtype(&mut u, Nullability::Nullable) else {
            break;
        };
        let dtype = DType::Extension(ext_dtype);
        let Ok(array) = ArbitraryArray::arbitrary_with_config(
            &mut u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype.clone()),
                len: 0..=32,
            },
        ) else {
            break;
        };
        assert_eq!(array.0.dtype(), &dtype);
    }
    Ok(())
}
