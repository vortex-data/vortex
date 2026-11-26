// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array conformance suites for [`Pivco`].

#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
use vortex_array::compute::conformance::cast::test_cast_conformance;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::mask::test_mask_conformance;
use vortex_array::compute::conformance::search_sorted::rstest_reuse::apply;
use vortex_array::compute::conformance::search_sorted::search_sorted_conformance;
use vortex_array::compute::conformance::search_sorted::*;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedArray;
use vortex_array::search_sorted::SearchSortedPrimitiveArray;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArraySlotsExt;
use vortex_zigzag::zigzag_encode;

use crate::Pivco;
use crate::PivcoArrayExt;
use crate::PivcoOptions;
use crate::bitmap_rans::BM_MARKER_RANS_BIT;
use crate::initialize;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    initialize(&session);
    vortex_zigzag::initialize(&session);
    session
});

fn encode_pivco(prim: &PrimitiveArray, opts: PivcoOptions) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    Pivco::encode_with_options(prim, opts, &mut ctx)
        .vortex_expect("Pivco::encode_with_options should succeed")
        .into_array()
}

fn slice_pivco(arr: &ArrayRef, start: usize, end: usize) -> ArrayRef {
    let sliced = arr
        .slice(start..end)
        .vortex_expect("slice should succeed")
        .optimize_ctx(&SESSION)
        .vortex_expect("optimize_ctx should succeed");
    assert!(
        sliced.is::<Pivco>(),
        "expected SliceReduce to produce PivcoArray, got {}",
        sliced.encoding_id()
    );
    sliced
}

fn encode_zigzag_pivco(signed: &PrimitiveArray, opts: PivcoOptions) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let zz = zigzag_encode(signed.as_view()).vortex_expect("zigzag_encode should succeed");
    let zz_prim = zz
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .vortex_expect("execute zigzag child");
    let child = Pivco::encode_with_options(&zz_prim, opts, &mut ctx)
        .vortex_expect("encode zigzag child")
        .into_array();
    ZigZag::try_new(child)
        .vortex_expect("ZigZag::try_new")
        .into_array()
}

fn make_skewed_u64(len: usize, seed: u64) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(seed);
    PrimitiveArray::from_iter((0..len).map(|_| {
        let d = rng.random_range(0..1000u32);
        if d < 900 {
            0u64
        } else if d < 980 {
            1u64
        } else {
            2u64
        }
    }))
}

fn make_mixed_nullable_u32(len: usize, seed: u64) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(seed);
    PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i % 7 == 0 {
            None
        } else {
            match i % 4 {
                0 => Some(0u32),
                1 => Some(rng.random_range(1..32u32)),
                2 => Some(50 + rng.random_range(0..100u32)),
                _ => Some(1_000 + rng.random_range(0..50_000u32)),
            }
        }
    }))
}

fn make_wide_ebits_u64(len: usize, seed: u64) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(seed);
    PrimitiveArray::from_iter((0..len).map(|i| match i % 5 {
        0 => 0u64,
        1 => rng.random_range(0..32u64),
        2 => (1u64 << 40) + rng.random_range(0..(1u64 << 20)),
        3 => (1u64 << 59) + rng.random_range(0..(1u64 << 58)),
        _ => (1u64 << 62) + rng.random_range(0..(1u64 << 61)),
    }))
}

/// Builds the conformance fixture matrix spanning all integer PTypes, codec modes (`PH`, `PHA`,
/// `OP-PHA`, `allow_flat = false`), extra-bit regimes, nullability variants, edge sizes, sliced
/// arrays, and `ZigZag(Pivco)`.
fn conformance_cases() -> Vec<(&'static str, ArrayRef)> {
    let mut cases: Vec<(&'static str, ArrayRef)> = Vec::new();

    // 1. All 8 integer PTypes across PH / PHA / OP-PHA / no-flat and !extra / extra bits
    cases.push((
        "u8_ph_no_extra",
        encode_pivco(
            &PrimitiveArray::from_iter((0..300u32).map(|i| (i % 19) as u8)),
            PivcoOptions::ph(),
        ),
    ));
    cases.push((
        "u8_pha_extra_no_flat",
        encode_pivco(
            &PrimitiveArray::from_iter((0..300u32).map(|i| ((i * 7) % 251) as u8)),
            PivcoOptions {
                allow_flat: false,
                ..PivcoOptions::pha()
            },
        ),
    ));
    cases.push((
        "u16_op_pha_extra",
        encode_pivco(
            &PrimitiveArray::from_iter((0..300u32).map(|i| ((i * 173) % 60_000) as u16)),
            PivcoOptions::op_pha(),
        ),
    ));
    cases.push((
        "u32_pha_nullable_mixed",
        encode_pivco(&make_mixed_nullable_u32(300, 11), PivcoOptions::pha()),
    ));
    cases.push((
        "u64_op_pha_wide_ebits_gt_57",
        encode_pivco(&make_wide_ebits_u64(300, 22), PivcoOptions::op_pha()),
    ));
    cases.push((
        "i8_nonneg_ph",
        encode_pivco(
            &PrimitiveArray::from_iter((0..200u32).map(|i| (i % 120) as i8)),
            PivcoOptions::ph(),
        ),
    ));
    cases.push((
        "i16_nonneg_pha",
        encode_pivco(
            &PrimitiveArray::from_iter((0..200u32).map(|i| ((i * 97) % 30_000) as i16)),
            PivcoOptions::pha(),
        ),
    ));
    cases.push((
        "i32_nonneg_op_pha",
        encode_pivco(
            &PrimitiveArray::from_iter((0..200u32).map(|i| ((i * 1_009) % 100_000) as i32)),
            PivcoOptions::op_pha(),
        ),
    ));
    cases.push((
        "i64_nonneg_pha_wide",
        encode_pivco(
            &PrimitiveArray::from_iter((0..200i64).map(|i| {
                if i % 3 == 0 {
                    i % 10
                } else {
                    (1i64 << 60) + i * 7919
                }
            })),
            PivcoOptions::pha(),
        ),
    ));

    // 2. Nullability variants: AllValid, AllInvalid
    cases.push((
        "nullable_all_valid",
        encode_pivco(
            &PrimitiveArray::new(
                Buffer::from_iter((0..256u32).map(|i| i % 50)),
                Validity::AllValid,
            ),
            PivcoOptions::pha(),
        ),
    ));
    cases.push((
        "nullable_all_invalid",
        encode_pivco(
            &PrimitiveArray::new(Buffer::from_iter(vec![0u32; 256]), Validity::AllInvalid),
            PivcoOptions::op_pha(),
        ),
    ));

    // 3. Edge sizes: empty (0) and single (1)
    cases.push((
        "empty_u32",
        encode_pivco(
            &PrimitiveArray::empty::<u32>(Nullability::NonNullable),
            PivcoOptions::pha(),
        ),
    ));
    cases.push((
        "empty_nullable_i64",
        encode_pivco(
            &PrimitiveArray::empty::<i64>(Nullability::Nullable),
            PivcoOptions::op_pha(),
        ),
    ));
    cases.push((
        "single_no_extra",
        encode_pivco(&PrimitiveArray::from_iter([7u32]), PivcoOptions::ph()),
    ));
    cases.push((
        "single_wide_extra",
        encode_pivco(
            &PrimitiveArray::from_iter([(1u64 << 61) + 123_456]),
            PivcoOptions::op_pha(),
        ),
    ));
    cases.push((
        "single_null",
        encode_pivco(
            &PrimitiveArray::from_option_iter([None::<i32>]),
            PivcoOptions::pha(),
        ),
    ));

    // 4. Multi-block (4097 with 32-lane rANS, 8500 OP-PHA nullable)
    let skewed_4097 = make_skewed_u64(4_097, 33);
    let rans32_4097 = encode_pivco(&skewed_4097, PivcoOptions::pha().with_block_shift(12));
    {
        let view = rans32_4097
            .as_opt::<Pivco>()
            .vortex_expect("must be PivcoArray");
        let tree_bytes = view.tree_payload().as_host();
        assert!(
            tree_bytes
                .as_ref()
                .iter()
                .any(|&b| (b & BM_MARKER_RANS_BIT) != 0),
            "skewed 4097 fixture should emit at least one 32-lane rANS node marker"
        );
    }
    cases.push(("multiblock_4097_32lane_rans", rans32_4097));

    let multiblock_8500 = encode_pivco(
        &make_mixed_nullable_u32(8_500, 44),
        PivcoOptions::op_pha().with_block_shift(12),
    );
    cases.push(("multiblock_8500_op_pha_nullable", multiblock_8500.clone()));

    // 5. Sliced PivcoArrays
    cases.push((
        "sliced_offset_gt_0",
        slice_pivco(&multiblock_8500, 137, 4_200),
    ));
    cases.push((
        "sliced_interior_multiblock",
        slice_pivco(&multiblock_8500, 2_500, 6_700),
    ));
    cases.push(("sliced_tail", slice_pivco(&multiblock_8500, 7_900, 8_500)));
    let wide_multi = encode_pivco(
        &make_wide_ebits_u64(5_000, 55),
        PivcoOptions::pha().with_block_shift(12),
    );
    cases.push((
        "sliced_wide_ebits_gt_57",
        slice_pivco(&wide_multi, 3_900, 4_950),
    ));

    // 6. ZigZag(Pivco) signed negative + positive
    let signed_i32 = PrimitiveArray::from_option_iter((0..500i32).map(|i| {
        if i % 11 == 0 {
            None
        } else if i % 2 == 0 {
            Some(-(i * 13))
        } else {
            Some(i * 17)
        }
    }));
    cases.push((
        "zigzag_pivco_i32_nullable",
        encode_zigzag_pivco(&signed_i32, PivcoOptions::pha()),
    ));
    let signed_i64 = PrimitiveArray::from_iter((0..300i64).map(|i| {
        if i % 3 == 0 {
            -(1i64 << 58) - i * 97
        } else {
            (1i64 << 57) + i * 31
        }
    }));
    cases.push((
        "zigzag_pivco_i64_wide",
        encode_zigzag_pivco(&signed_i64, PivcoOptions::op_pha()),
    ));

    cases
}

#[test]
fn conformance_consistency_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_array_consistency(&array, &mut ctx);
    }
}

#[test]
fn conformance_take_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_take_conformance(&array, &mut ctx);
    }
}

#[test]
fn conformance_filter_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_filter_conformance(&array, &mut ctx);
    }
}

#[test]
fn conformance_mask_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_mask_conformance(&array, &mut ctx);
    }
}

#[test]
fn conformance_cast_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_cast_conformance(&array, &mut ctx);
    }
}

#[test]
fn conformance_binary_numeric_suite() {
    let mut ctx = SESSION.create_execution_ctx();
    for (name, array) in conformance_cases() {
        assert!(!name.is_empty());
        test_binary_numeric_array(&array, &mut ctx);
    }
}

#[apply(search_sorted_conformance)]
fn conformance_search_sorted_pivco_template(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let prim = array.execute::<PrimitiveArray>(&mut ctx)?;
    for opts in [
        PivcoOptions::ph(),
        PivcoOptions::pha(),
        PivcoOptions::op_pha(),
    ] {
        let pivco = Pivco::encode_with_options(&prim, opts, &mut ctx)?.into_array();

        let res_prim =
            SearchSortedPrimitiveArray::<i32>::new(&pivco, &mut ctx).search_sorted(&value, side)?;
        assert_eq!(res_prim, expected);

        let res_scalar =
            SearchSortedArray::new(&pivco, &mut ctx).search_sorted(&Scalar::from(value), side)?;
        assert_eq!(res_scalar, expected);
    }
    Ok(())
}

#[rstest]
#[case::ph(PivcoOptions::ph().with_block_shift(10))]
#[case::pha(PivcoOptions::pha().with_block_shift(10))]
#[case::op_pha(PivcoOptions::op_pha().with_block_shift(10))]
fn conformance_search_sorted_multiblock_and_sliced(#[case] opts: PivcoOptions) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let non_null_vals: Vec<u32> = (0..2_500u32).map(|i| (i / 3) * (i / 150 + 1)).collect();
    let non_null_prim = PrimitiveArray::from_iter(non_null_vals);
    let non_null_pivco = Pivco::encode_with_options(&non_null_prim, opts, &mut ctx)?.into_array();
    let non_null_sliced = slice_pivco(&non_null_pivco, 317, 2_199);
    let non_null_ref_sliced = non_null_prim.clone().into_array().slice(317..2_199)?;

    let nullable_prim = PrimitiveArray::from_option_iter((0..2_500u32).map(|i| {
        if i < 150 {
            None
        } else {
            Some(((i - 150) / 4) * 7)
        }
    }));
    let nullable_pivco = Pivco::encode_with_options(&nullable_prim, opts, &mut ctx)?.into_array();
    let nullable_sliced = slice_pivco(&nullable_pivco, 90, 2_200);
    let nullable_ref_sliced = nullable_prim.clone().into_array().slice(90..2_200)?;

    let queries = [0u32, 1, 7, 14, 31, 32, 100, 500, 2_000, 10_000, 999_999];
    for (encoded, reference) in [
        (&non_null_pivco, non_null_prim.into_array()),
        (&non_null_sliced, non_null_ref_sliced),
        (&nullable_pivco, nullable_prim.into_array()),
        (&nullable_sliced, nullable_ref_sliced),
    ] {
        for &q in &queries {
            for side in [SearchSortedSide::Left, SearchSortedSide::Right] {
                let expected_raw = SearchSortedPrimitiveArray::<u32>::new(&reference, &mut ctx)
                    .search_sorted(&q, side)?;
                let actual_raw = SearchSortedPrimitiveArray::<u32>::new(encoded, &mut ctx)
                    .search_sorted(&q, side)?;
                assert_eq!(actual_raw, expected_raw);

                let expected_opt = SearchSortedPrimitiveArray::<u32>::new(&reference, &mut ctx)
                    .search_sorted(&Some(q), side)?;
                let actual_opt = SearchSortedPrimitiveArray::<u32>::new(encoded, &mut ctx)
                    .search_sorted(&Some(q), side)?;
                let actual_scalar = SearchSortedArray::new(encoded, &mut ctx)
                    .search_sorted(&Scalar::from(Some(q)), side)?;
                assert_eq!(actual_opt, expected_opt);
                assert_eq!(actual_scalar, expected_opt);
            }
        }
    }
    Ok(())
}
