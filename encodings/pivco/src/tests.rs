// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::LazyLock;

use prost::Message as _;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArraySlotsExt;
use vortex_zigzag::zigzag_encode;

use crate::BLOCK_HEADER_BYTES_NO_EXTRA;
use crate::BLOCK_HEADER_BYTES_WITH_EXTRA;
use crate::MAX_SYMS;
use crate::Pivco;
use crate::PivcoArrayExt;
use crate::PivcoBlockOffset;
use crate::PivcoData;
use crate::PivcoMetadata;
use crate::PivcoOptions;
use crate::PivcoPlugin;
use crate::PivcoTree;
use crate::PivcoTreeMode;
use crate::bitmap_rans::BM_LANES;
use crate::bitmap_rans::BM_MARKER_ALL_ONE;
use crate::bitmap_rans::BM_MARKER_ALL_ZERO;
use crate::bitmap_rans::BM_MARKER_RANS_BIT;
use crate::bitmap_rans::BM_MARKER_RAW;
use crate::bitmap_rans::BM_MARKER_SPARSE_ONE;
use crate::bitmap_rans::BM_MARKER_SPARSE_ZERO;
use crate::bitmap_rans::DenseBitmap;
use crate::bitmap_rans::NodeBitmap;
use crate::bitmap_rans::compute_bernoulli_freqs;
use crate::bitmap_rans::decode_static_byte_rans;
use crate::bitmap_rans::encode_static_byte_rans_into;
use crate::bitmap_rans::write_node_bitmap;
use crate::block_header_stride;
use crate::codec::DecodeWorkspace;
use crate::codec::block_len_for;
use crate::estimate_compressed_bits;
use crate::initialize;
use crate::simd::MERGE_OVERREAD;
use crate::simd::X86_MERGE_SHUF0;
use crate::simd::X86_MERGE_SHUF1;
use crate::simd::merge_cst_cst;
use crate::simd::merge_cst_cst_scalar;
use crate::simd::merge_cst_vec;
use crate::simd::merge_cst_vec_in_place;
use crate::simd::merge_cst_vec_scalar;
use crate::simd::merge_vec_cst;
use crate::simd::merge_vec_cst_in_place;
use crate::simd::merge_vec_cst_scalar;
use crate::simd::merge_vec_vec;
use crate::simd::merge_vec_vec_in_place_left;
use crate::simd::merge_vec_vec_in_place_right;
use crate::simd::merge_vec_vec_scalar;
use crate::simd::pack_64_bytes01;
use crate::simd::pack_64_bytes01_scalar;
use crate::simd::pack_flat_codes;
use crate::simd::pack_flat_codes_scalar;
use crate::simd::partition_ranks_by_pivot;
use crate::simd::partition_ranks_scalar;
use crate::simd::popcount_bits;
use crate::simd::popcount_bits_scalar;
use crate::simd::unpack_flat_codes;
use crate::simd::unpack_flat_codes_scalar;
use crate::tree::ChildSpec;
use crate::tree::PivcoNode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    initialize(&session);
    vortex_zigzag::initialize(&session);
    session
});

fn slice_pivco(arr: &ArrayRef, start: usize, end: usize) -> ArrayRef {
    let sliced = arr
        .slice(start..end)
        .vortex_expect("slice should succeed")
        .optimize_ctx(&SESSION)
        .vortex_expect("optimize_ctx should succeed");
    assert!(sliced.is::<Pivco>());
    sliced
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

// ============================================================================
// Codec End-to-End Roundtrip, Extreme Bounds, and Point Access
// ============================================================================

#[rstest]
#[case::ph_b10(PivcoOptions::ph().with_block_shift(10))]
#[case::ph_b12(PivcoOptions::ph().with_block_shift(12))]
#[case::pha_b12(PivcoOptions::pha().with_block_shift(12))]
#[case::pha_b13(PivcoOptions::pha().with_block_shift(13))]
#[case::pha_no_flat(PivcoOptions { allow_flat: false, ..PivcoOptions::pha() })]
#[case::op_pha_b12(PivcoOptions::op_pha().with_block_shift(12))]
fn roundtrip_skewed_wide_and_zigzag(#[case] opts: PivcoOptions) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(42);
    let n = 25_000usize;

    // 1. Skewed nullable `{0, 1, 2}` u64 stream
    let gt = PrimitiveArray::from_option_iter((0..n).map(|_| {
        let d = rng.random_range(0..1000u32);
        if d < 200 {
            None
        } else if d < 850 {
            Some(0u64)
        } else if d < 990 {
            Some(1u64)
        } else {
            Some(2u64)
        }
    }));
    let enc_gt = Pivco::encode_with_options(&gt, opts, &mut ctx)?.into_array();
    assert_arrays_eq!(&enc_gt, &gt.clone().into_array(), &mut ctx);

    // 2. Extra-bits i32 stream + nested slices + serde + filter (sparse & dense) + take + scalar_at
    let pl = PrimitiveArray::from_iter((0..n).map(|i| match i % 3 {
        0 => 0i32,
        1 => 45 + rng.random_range(0..35i32),
        _ => 750 + rng.random_range(0..350i32),
    }));
    let enc_pl = Pivco::encode_with_options(&pl, opts, &mut ctx)?.into_array();
    assert_arrays_eq!(&enc_pl, &pl.clone().into_array(), &mut ctx);

    let s1 = slice_pivco(&enc_pl, 137, 19_800);
    let exp_s1 = pl.clone().into_array().slice(137..19_800)?;
    assert_arrays_eq!(&s1, &exp_s1, &mut ctx);
    let s2 = slice_pivco(&s1, 4_200, 15_111);
    let exp_s2 = exp_s1.slice(4_200..15_111)?;
    assert_arrays_eq!(&s2, &exp_s2, &mut ctx);

    let ser = PivcoPlugin.serialize(&s2, &SESSION)?.expect("serializable");
    let ser_buffers: Vec<BufferHandle> = ser
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect();
    let deser = PivcoPlugin.deserialize(
        ArrayDeserialization::new(
            ser.serialized_id,
            s2.dtype(),
            s2.len(),
            &ser.metadata,
            &ser_buffers,
            &ser.children,
        ),
        &SESSION,
    )?;
    assert_arrays_eq!(&deser, &exp_s2, &mut ctx);

    // Ultra-sparse filter (hits `<= 4` per block scalar_at fast path), 1% filter, and >1/8 high-selectivity filter
    for step in [2_000usize, 97, 3] {
        let mask = Mask::from_buffer(BitBuffer::from_iter((0..s2.len()).map(|i| i % step == 3)));
        assert_arrays_eq!(&s2.filter(mask.clone())?, &exp_s2.filter(mask)?, &mut ctx);
    }

    // Take: ultra-sparse (<= 4/block), ordered, and scattered
    let s2_len = u32::try_from(s2.len())?;
    for idx in [
        PrimitiveArray::from_iter([0u32, 13, 5_000, s2_len - 1]).into_array(),
        PrimitiveArray::from_iter((0..80u32).map(|k| (k * 131) % s2_len)).into_array(),
    ] {
        assert_arrays_eq!(&s2.take(idx.clone())?, &exp_s2.take(idx)?, &mut ctx);
    }

    for spot in [0, 1, 63, 64, 1023, 1024, 4095, 4096, s2.len() - 1] {
        if spot < s2.len() {
            assert_eq!(
                s2.execute_scalar(spot, &mut ctx)?,
                exp_s2.execute_scalar(spot, &mut ctx)?
            );
        }
    }

    // 3. Extreme integer bounds (`u64::MAX`, `ebits > 57`, and `ZigZag(Pivco)` with `i64::MIN`)
    let u64_vals: Vec<u64> = (0..1_200usize)
        .map(|i| match i % 8 {
            0 => 0,
            1 => 1,
            2 => 31,
            3 => 32,
            4 => (1u64 << 57) - 1,
            5 => (1u64 << 58) + (i as u64),
            6 => u64::MAX - 1,
            _ => u64::MAX,
        })
        .collect();
    let u64_prim = PrimitiveArray::from_iter(u64_vals);
    let u64_enc = Pivco::encode_with_options(&u64_prim, opts, &mut ctx)?.into_array();
    let u64_exp = u64_prim.into_array();
    assert_arrays_eq!(&u64_enc, &u64_exp, &mut ctx);

    let i64_prim = PrimitiveArray::from_option_iter((0..1_200usize).map(|i| match i % 9 {
        0 => None,
        1 => Some(0i64),
        2 => Some(-1),
        3 => Some(1),
        4 => Some(i64::MIN),
        5 => Some(i64::MIN + 1),
        6 => Some(i64::MAX - 1),
        7 => Some(i64::MAX),
        _ => Some(-(1i64 << 57) + (i as i64)),
    }));
    let zz = zigzag_encode(i64_prim.as_view())?;
    let zz_unsigned = zz.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let zz_child = Pivco::encode_with_options(&zz_unsigned, opts, &mut ctx)?.into_array();
    let zz_pivco = ZigZag::try_new(zz_child)?.into_array();
    assert_arrays_eq!(&zz_pivco, &i64_prim.into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::pha(PivcoOptions::pha())]
#[case::op_pha(PivcoOptions::op_pha())]
fn spine_compare_pushdown_matches_canonical(#[case] opts: PivcoOptions) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(2026);
    let n = 15_000usize;

    let arr = PrimitiveArray::from_option_iter((0..n).map(|_| {
        let u = rng.random_range(0..1000u32);
        if u < 100 {
            None
        } else if u < 600 {
            Some(0i32)
        } else if u < 800 {
            Some(rng.random_range(1..32i32))
        } else {
            Some(rng.random_range(64..1024i32))
        }
    }));

    let expected_arr = arr.clone().into_array();
    let pivco_arr = Pivco::encode_with_options(&arr, opts, &mut ctx)?.into_array();
    let sliced_pivco = slice_pivco(&pivco_arr, 250, 13_500);
    let sliced_exp = expected_arr.slice(250..13_500)?;

    let ops = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];
    for c in [-5i32, 0, 1, 7, 31, 32, 48, 64, 500, 2000] {
        for op in ops {
            let rhs_p = ConstantArray::new(Scalar::from(c), pivco_arr.len()).into_array();
            assert_arrays_eq!(
                &pivco_arr.clone().compare(rhs_p.clone(), op)?,
                &expected_arr.clone().compare(rhs_p, op)?,
                &mut ctx
            );

            let rhs_s = ConstantArray::new(Scalar::from(c), sliced_pivco.len()).into_array();
            assert_arrays_eq!(
                &sliced_pivco.clone().compare(rhs_s.clone(), op)?,
                &sliced_exp.clone().compare(rhs_s, op)?,
                &mut ctx
            );
        }
    }
    Ok(())
}

#[rstest]
#[case::ph_b10(PivcoOptions::ph().with_block_shift(10))]
#[case::pha_b12(PivcoOptions::pha().with_block_shift(12))]
#[case::op_pha_b12(PivcoOptions::op_pha().with_block_shift(12))]
fn scalar_at_matches_execute_across_all_node_modes_and_slices(
    #[case] opts: PivcoOptions,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0xA5A5_0001);

    // Multi-block array exercising AllZero, AllOne, SparseOne, SparseZero, 32-lane rANS, Raw,
    // Flat, nulls, < 32, 32..2^57, and > 2^59 (WideBitReader).
    let mixed = PrimitiveArray::from_option_iter((0..10_000usize).map(|idx| {
        if idx < 4_096 {
            if idx % 257 == 0 {
                Some((1u64 << 60) + (idx as u64) * 19)
            } else {
                Some(0u64)
            }
        } else if idx < 8_192 {
            if idx % 19 == 0 {
                None
            } else if rng.random_range(0..100u32) < 90 {
                Some(0u64)
            } else {
                Some(rng.random_range(1..2048u64))
            }
        } else {
            let bucket = rng.random_range(0..40u64);
            if bucket < 20 {
                Some(bucket)
            } else if bucket < 38 {
                Some(32 + rng.random_range(0..100_000u64))
            } else {
                Some((1u64 << 61) + rng.random_range(0..1_000_000u64))
            }
        }
    }));
    let exp_mixed = mixed.clone().into_array();
    let enc_mixed = Pivco::encode_with_options(&mixed, opts, &mut ctx)?.into_array();
    assert_arrays_eq!(&enc_mixed, &exp_mixed, &mut ctx);

    let sliced = slice_pivco(&enc_mixed, 777, 9_555);
    let exp_sliced = exp_mixed.slice(777..9_555)?;
    for pos in 0..exp_sliced.len() {
        assert_eq!(
            sliced.execute_scalar(pos, &mut ctx)?,
            exp_sliced.execute_scalar(pos, &mut ctx)?,
            "scalar_at mismatch at sliced pos={pos}"
        );
    }

    Ok(())
}

// ============================================================================
// Wire Layout, Alignment, Windowed Slice, L1 Workspace, and Negative Validation
// ============================================================================

#[test]
fn three_buffer_no_extra_layout_and_slice() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let small = make_skewed_u64(5_000, 701);
    let enc_small =
        Pivco::encode_with_options(&small, PivcoOptions::pha().with_block_shift(10), &mut ctx)?;
    let view_s = enc_small.as_view();
    assert_eq!(Pivco::nbuffers(view_s), 3);
    assert_eq!(
        Pivco::buffer_name(view_s, 0).as_deref(),
        Some("block_headers")
    );
    assert_eq!(
        Pivco::buffer_name(view_s, 1).as_deref(),
        Some("tree_payload")
    );
    assert_eq!(Pivco::buffer_name(view_s, 2).as_deref(), Some("raw_bits"));
    assert!(!view_s.has_extra_bits());
    assert_eq!(
        view_s.block_headers().len(),
        6 * BLOCK_HEADER_BYTES_NO_EXTRA
    );
    assert_eq!(view_s.raw_bits().len(), 0);
    assert_eq!(view_s.last_block_len(), 5_000 - 4 * 1_024);

    let sliced_small = slice_pivco(&enc_small.into_array(), 1_500, 3_500);
    let sv = sliced_small.as_opt::<Pivco>().vortex_expect("PivcoArray");
    assert_eq!(sv.block_headers().len(), 4 * BLOCK_HEADER_BYTES_NO_EXTRA);
    assert_eq!(sv.raw_bits().len(), 0);
    assert_eq!(sv.last_block_len(), 1_024);
    assert!(sv.tree_base_offset() > 0);
    assert_eq!(sv.raw_base_word_offset(), 0);
    assert_arrays_eq!(
        &sliced_small,
        &small.into_array().slice(1_500..3_500)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn three_buffer_extra_bits_sliced_serde_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let wide = make_wide_ebits_u64(5_000, 702);
    let enc_wide =
        Pivco::encode_with_options(&wide, PivcoOptions::pha().with_block_shift(10), &mut ctx)?;
    let view_w = enc_wide.as_view();
    assert!(view_w.has_extra_bits());
    assert_eq!(
        view_w.block_headers().len(),
        6 * BLOCK_HEADER_BYTES_WITH_EXTRA
    );
    assert!(view_w.raw_bits().is_aligned_to(Alignment::of::<u64>()));
    assert_eq!(view_w.raw_bits().len() % 8, 0);

    for (lo, hi, expected_last_len) in [
        (1_100usize, 3_200usize, 1_024u16),
        (3_100usize, 5_000usize, (5_000 - 4 * 1_024) as u16),
        (4_500usize, 4_500usize, 0u16),
    ] {
        let sliced = if lo == hi {
            Pivco::slice(enc_wide.as_view(), lo..hi)?.vortex_expect("empty SliceReduce")
        } else {
            slice_pivco(&enc_wide.clone().into_array(), lo, hi)
        };
        let sv = sliced.as_opt::<Pivco>().vortex_expect("PivcoArray");
        assert_eq!(sv.last_block_len(), expected_last_len);
        let ser = PivcoPlugin
            .serialize(&sliced, &SESSION)?
            .vortex_expect("ser");
        assert_eq!(ser.buffers.len(), 3);
        let bufs: Vec<BufferHandle> = ser
            .buffers
            .into_iter()
            .map(BufferHandle::new_host)
            .collect();
        let deser = PivcoPlugin.deserialize(
            ArrayDeserialization::new(
                ser.serialized_id,
                sliced.dtype(),
                sliced.len(),
                &ser.metadata,
                &bufs,
                &ser.children,
            ),
            &SESSION,
        )?;
        assert_arrays_eq!(&deser, &wide.clone().into_array().slice(lo..hi)?, &mut ctx);
    }

    Ok(())
}

#[test]
fn zero_slack_tail_extra_bits_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0x7A11_0000);

    for len in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1_024, 1_025] {
        for &(shift_bits, trailing_zero) in &[
            (5u32, false),
            (13, true),
            (31, false),
            (57, false),
            (57, true),
            (59, false),
            (62, false),
            (62, true),
        ] {
            let max_mask = if shift_bits >= 63 {
                u64::MAX
            } else {
                (1u64 << shift_bits) - 1
            };
            let vals: Vec<u64> = (0..len)
                .map(|i| {
                    if trailing_zero && i + 1 == len {
                        0u64
                    } else {
                        (1u64 << shift_bits) | (rng.random::<u64>() & max_mask)
                    }
                })
                .collect();
            let prim = PrimitiveArray::from_iter(vals);
            let pivco = Pivco::encode_with_options(
                &prim,
                PivcoOptions::pha().with_block_shift(10),
                &mut ctx,
            )?
            .into_array();
            assert_arrays_eq!(&pivco, &prim.clone().into_array(), &mut ctx);

            for idx in [0usize, len.saturating_sub(2), len - 1] {
                assert_eq!(
                    pivco.execute_scalar(idx, &mut ctx)?,
                    prim.clone().into_array().execute_scalar(idx, &mut ctx)?
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum HeaderCorruption {
    TooFewBuffers,
    TooManyBuffers,
    NonNullableWithChild,
    TwoChildren,
    MisalignedRawBits,
    MisalignedTreePayload,
    TruncatedTreePayload,
    TruncatedRawBits,
    UnalignedTreeBase,
    PrecedingTreeBase,
    PrecedingRawBase,
    NonMonotonicRawHeader,
    UnalignedTreeHeader,
    LastBlockLenTooSmall,
    LastBlockLenZero,
    NonEmptyRawWithoutExtraBits,
}

#[rstest]
#[case::too_few_buffers(HeaderCorruption::TooFewBuffers)]
#[case::too_many_buffers(HeaderCorruption::TooManyBuffers)]
#[case::non_nullable_child(HeaderCorruption::NonNullableWithChild)]
#[case::two_children(HeaderCorruption::TwoChildren)]
#[case::misaligned_raw(HeaderCorruption::MisalignedRawBits)]
#[case::misaligned_tree(HeaderCorruption::MisalignedTreePayload)]
#[case::truncated_tree(HeaderCorruption::TruncatedTreePayload)]
#[case::truncated_raw(HeaderCorruption::TruncatedRawBits)]
#[case::unaligned_tree_base(HeaderCorruption::UnalignedTreeBase)]
#[case::preceding_tree_base(HeaderCorruption::PrecedingTreeBase)]
#[case::preceding_raw_base(HeaderCorruption::PrecedingRawBase)]
#[case::non_monotonic_raw_hdr(HeaderCorruption::NonMonotonicRawHeader)]
#[case::unaligned_tree_hdr(HeaderCorruption::UnalignedTreeHeader)]
#[case::last_len_small(HeaderCorruption::LastBlockLenTooSmall)]
#[case::last_len_zero(HeaderCorruption::LastBlockLenZero)]
#[case::bogus_raw_no_extra(HeaderCorruption::NonEmptyRawWithoutExtraBits)]
fn rejects_corrupt_metadata_and_headers(#[case] case: HeaderCorruption) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let arr = make_wide_ebits_u64(2_500, 909);
    let pivco =
        Pivco::encode_with_options(&arr, PivcoOptions::pha().with_block_shift(10), &mut ctx)?
            .into_array();
    let ser = PivcoPlugin
        .serialize(&pivco, &SESSION)?
        .vortex_expect("ser");
    let mut meta = PivcoMetadata::decode(ser.metadata.as_slice())?;
    let mut bufs: Vec<BufferHandle> = ser
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect();
    let mut children = ser.children.clone();
    let mut dtype = pivco.dtype().clone();
    let mut len = pivco.len();

    let make_unaligned = |src: &BufferHandle| -> BufferHandle {
        let host = src.as_host();
        let mut tmp = ByteBufferMut::with_capacity_aligned(host.len() + 1, Alignment::of::<u8>());
        tmp.push(0u8);
        tmp.extend_from_slice(host.as_ref());
        BufferHandle::new_host(tmp.freeze().slice_unaligned(1..))
    };

    match case {
        HeaderCorruption::TooFewBuffers => {
            bufs.pop();
        }
        HeaderCorruption::TooManyBuffers => {
            bufs.push(bufs[2].clone());
        }
        HeaderCorruption::NonNullableWithChild => {
            children.push(ConstantArray::new(true, len).into_array());
        }
        HeaderCorruption::TwoChildren => {
            dtype = dtype.as_nullable();
            let c = ConstantArray::new(true, len).into_array();
            children = vec![c.clone(), c];
        }
        HeaderCorruption::MisalignedRawBits => {
            bufs[2] = make_unaligned(&bufs[2]);
            assert!(!bufs[2].is_aligned_to(Alignment::of::<u64>()));
        }
        HeaderCorruption::MisalignedTreePayload => {
            bufs[1] = make_unaligned(&bufs[1]);
            assert!(!bufs[1].is_aligned_to(Alignment::of::<u32>()));
        }
        HeaderCorruption::TruncatedTreePayload => {
            bufs[1] = BufferHandle::new_host(
                bufs[1]
                    .as_host()
                    .slice_with_alignment(0..bufs[1].len() - 4, Alignment::of::<u32>()),
            );
        }
        HeaderCorruption::TruncatedRawBits => {
            bufs[2] = BufferHandle::new_host(
                bufs[2]
                    .as_host()
                    .slice_with_alignment(0..bufs[2].len() - 8, Alignment::of::<u64>()),
            );
        }
        HeaderCorruption::UnalignedTreeBase => meta.tree_base_offset = 1,
        HeaderCorruption::PrecedingTreeBase => meta.tree_base_offset = 4,
        HeaderCorruption::PrecedingRawBase => meta.raw_base_word_offset = 1,
        HeaderCorruption::NonMonotonicRawHeader => {
            let mut h = ByteBufferMut::with_capacity_aligned(bufs[0].len(), Alignment::of::<u32>());
            h.extend_from_slice(bufs[0].as_host().as_ref());
            h[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
            bufs[0] = BufferHandle::new_host(h.freeze());
        }
        HeaderCorruption::UnalignedTreeHeader => {
            let mut h = ByteBufferMut::with_capacity_aligned(bufs[0].len(), Alignment::of::<u32>());
            h.extend_from_slice(bufs[0].as_host().as_ref());
            let b1 = u32::from_le_bytes([h[8], h[9], h[10], h[11]]);
            h[8..12].copy_from_slice(&(b1 + 1).to_le_bytes());
            bufs[0] = BufferHandle::new_host(h.freeze());
        }
        HeaderCorruption::LastBlockLenTooSmall => meta.last_block_len -= 1,
        HeaderCorruption::LastBlockLenZero => meta.last_block_len = 0,
        HeaderCorruption::NonEmptyRawWithoutExtraBits => {
            let small_arr = PrimitiveArray::from_iter((0..1_200u32).map(|i| i % 5));
            let small_pivco = Pivco::encode_with_options(
                &small_arr,
                PivcoOptions::ph().with_block_shift(10),
                &mut ctx,
            )?
            .into_array();
            let s_ser = PivcoPlugin
                .serialize(&small_pivco, &SESSION)?
                .vortex_expect("ser");
            meta = PivcoMetadata::decode(s_ser.metadata.as_slice())?;
            bufs = s_ser
                .buffers
                .into_iter()
                .map(BufferHandle::new_host)
                .collect();
            bufs[2] = BufferHandle::new_host(ByteBuffer::copy_from_aligned(
                [0u8; 8],
                Alignment::of::<u64>(),
            ));
            children = s_ser.children;
            dtype = small_pivco.dtype().clone();
            len = small_pivco.len();
        }
    }

    assert!(
        PivcoPlugin
            .deserialize(
                ArrayDeserialization::new(
                    ser.serialized_id,
                    &dtype,
                    len,
                    &meta.encode_to_vec(),
                    &bufs,
                    &children,
                ),
                &SESSION,
            )
            .is_err()
    );
    Ok(())
}

#[derive(Default)]
struct AlignmentAuditCounts {
    sparse_nodes: usize,
    rans32_nodes: usize,
    flat_nodes: usize,
}

fn audit_node_pointer_alignment(
    tree: &PivcoTree,
    node_idx: u16,
    n: usize,
    wire: &[u8],
    cursor: &mut usize,
    counts: &mut AlignmentAuditCounts,
) -> VortexResult<()> {
    if n == 0 {
        return Ok(());
    }
    match &tree.nodes[usize::from(node_idx)] {
        PivcoNode::SingleLeaf { .. } => Ok(()),
        PivcoNode::Flat { depth, .. } => {
            counts.flat_nodes += 1;
            let nbytes = (n * usize::from(*depth)).div_ceil(8);
            assert!(*cursor + nbytes <= wire.len());
            *cursor += nbytes;
            Ok(())
        }
        PivcoNode::Internal {
            left_child,
            right_child,
            ..
        } => {
            let marker_pos = *cursor;
            let marker = wire[marker_pos];
            match marker {
                BM_MARKER_ALL_ZERO | BM_MARKER_ALL_ONE | BM_MARKER_RAW => {}
                BM_MARKER_SPARSE_ONE | BM_MARKER_SPARSE_ZERO => {
                    counts.sparse_nodes += 1;
                    let count_pos = (marker_pos + 2) & !1;
                    assert_eq!((wire[count_pos..].as_ptr() as usize) % 2, 0);
                    assert_eq!((wire[count_pos + 2..].as_ptr() as usize) % 2, 0);
                }
                code if (code & BM_MARKER_RANS_BIT) != 0 => {
                    counts.rans32_nodes += 1;
                    let states_pos = (marker_pos + 6) & !3;
                    let coded_len_pos = states_pos - 2;
                    assert_eq!((wire[coded_len_pos..].as_ptr() as usize) % 2, 0);
                    assert_eq!((wire[states_pos..].as_ptr() as usize) % 4, 0);
                    assert_eq!((wire[states_pos + BM_LANES * 4..].as_ptr() as usize) % 2, 0);
                }
                other => vortex_bail!("unexpected bitmap marker {other:#x}"),
            }

            let mut scratch = vec![MaybeUninit::<u8>::uninit(); n.div_ceil(8)];
            let bm = NodeBitmap::read(wire, cursor, n, &mut scratch)?;
            let k_right = bm.count1(&scratch, n);
            let k_left = n - k_right;

            if let ChildSpec::Node(l_idx) = left_child {
                let lb = usize::from(u16::from_le_bytes([wire[*cursor], wire[*cursor + 1]]));
                *cursor += 2;
                let left_start = *cursor;
                if k_left > 0 {
                    audit_node_pointer_alignment(tree, *l_idx, k_left, wire, cursor, counts)?;
                }
                assert_eq!(*cursor - left_start, lb);
            }
            if let ChildSpec::Node(r_idx) = right_child
                && k_right > 0
            {
                audit_node_pointer_alignment(tree, *r_idx, k_right, wire, cursor, counts)?;
            }
            Ok(())
        }
    }
}

#[test]
fn natural_pointer_alignment_in_tree_payload() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0x6900_A116);
    let len = 10_000usize;

    let data = PrimitiveArray::from_iter((0..len).map(|i| {
        if i < 4_096 {
            if rng.random_range(0..1000u32) < 5 {
                rng.random_range(1..17u32)
            } else {
                0u32
            }
        } else if rng.random_range(0..1000u32) < 900 {
            0u32
        } else {
            rng.random_range(1..17u32)
        }
    }));

    let mut counts = AlignmentAuditCounts::default();
    for opts in [
        PivcoOptions::ph().with_block_shift(12),
        PivcoOptions::pha().with_block_shift(12),
        PivcoOptions::op_pha().with_block_shift(12),
    ] {
        let encoded = Pivco::encode_with_options(&data, opts, &mut ctx)?.into_array();
        let sliced = slice_pivco(&encoded, 1_500, 9_500);
        for arr in [&encoded, &sliced] {
            let view = arr.as_opt::<Pivco>().vortex_expect("PivcoArray");
            let hdr_host = view.block_headers().as_host();
            let tree_host = view.tree_payload().as_host();
            assert_eq!((tree_host.as_ptr() as usize) % 4, 0);

            let block_values = view.block_values();
            let num_blocks = (usize::from(view.offset()) + view.len()).div_ceil(block_values);
            let tree_base = view.tree_base_offset() as usize;
            for b in 0..num_blocks {
                let start_hdr =
                    PivcoBlockOffset::read_at(hdr_host.as_ref(), b, view.has_extra_bits());
                let end_hdr =
                    PivcoBlockOffset::read_at(hdr_host.as_ref(), b + 1, view.has_extra_bits());
                assert_eq!(start_hdr.tree_byte_offset % 4, 0);
                assert_eq!(end_hdr.tree_byte_offset % 4, 0);
                let rel_start = (start_hdr.tree_byte_offset as usize) - tree_base;
                let rel_end = (end_hdr.tree_byte_offset as usize) - tree_base;
                let block_wire = &tree_host.as_ref()[rel_start..rel_end];
                assert_eq!((block_wire.as_ptr() as usize) % 4, 0);
                let block_len = block_len_for(b, num_blocks, block_values, view.last_block_len());
                let mut cursor = 0usize;
                audit_node_pointer_alignment(
                    view.tree(),
                    0,
                    block_len,
                    block_wire,
                    &mut cursor,
                    &mut counts,
                )?;
                assert!(cursor <= block_wire.len() && block_wire.len() - cursor < 4);
            }
        }
    }

    assert!(counts.sparse_nodes > 0, "expected SPARSE nodes audited");
    assert!(
        counts.rans32_nodes > 0,
        "expected 32-lane RANS nodes audited"
    );
    assert!(counts.flat_nodes > 0, "expected Flat nodes audited");
    Ok(())
}

#[test]
fn windowed_untrimmed_slice_matches_host_trimmed_slice() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let cases = [make_skewed_u64(5_000, 801), make_wide_ebits_u64(5_000, 802)];

    for prim in cases {
        let encoded = Pivco::encode_with_options(
            &prim,
            PivcoOptions::op_pha().with_block_shift(10),
            &mut ctx,
        )?
        .into_array();
        let view = encoded.as_opt::<Pivco>().vortex_expect("PivcoArray");
        let lo = 1_317usize;
        let hi = 3_812usize;
        let host_trimmed = slice_pivco(&encoded, lo, hi);

        let block_shift = view.block_shift();
        let block_values = 1usize << block_shift;
        let first_block = lo >> block_shift;
        let end_block = hi.div_ceil(block_values);
        let stride = block_header_stride(view.has_extra_bits());
        let untrimmed_headers = view
            .block_headers()
            .slice(first_block * stride..(end_block + 1) * stride);
        let untrimmed_data = PivcoData::try_new(
            block_shift,
            (lo & (block_values - 1)) as u16,
            block_values as u16,
            view.tree_base_offset(),
            view.raw_base_word_offset(),
            Arc::clone(view.tree()),
            untrimmed_headers,
            view.tree_payload().clone(),
            view.raw_bits().clone(),
        )?;
        let untrimmed = Pivco::try_new(
            view.dtype().clone(),
            hi - lo,
            untrimmed_data,
            view.pivco_validity().slice(lo..hi)?,
        )?
        .into_array();

        assert_arrays_eq!(&untrimmed, &host_trimmed, &mut ctx);
        for idx in [0, 1, 706, 707, 1_730, 1_731, hi - lo - 1] {
            assert_eq!(
                untrimmed.execute_scalar(idx, &mut ctx)?,
                host_trimmed.execute_scalar(idx, &mut ctx)?
            );
        }
        let take_idx =
            PrimitiveArray::from_iter((0..64u32).map(|k| ((k * 137) as usize % (hi - lo)) as u32))
                .into_array();
        assert_arrays_eq!(
            &untrimmed.take(take_idx.clone())?,
            &host_trimmed.take(take_idx)?,
            &mut ctx
        );
        let mask = Mask::from_buffer(BitBuffer::from_iter((0..(hi - lo)).map(|i| i % 29 == 5)));
        assert_arrays_eq!(
            &untrimmed.filter(mask.clone())?,
            &host_trimmed.filter(mask)?,
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn workspace_stays_l1_resident_and_decodes_full_150_symbol_trees() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let mut all_syms = Vec::with_capacity(8_192);
    for i in 0..8_192u64 {
        let sym = (i % 150) as usize;
        let val = if sym < 32 {
            sym as u64
        } else {
            let rel = (sym - 32) as u32;
            let k = 5 + (rel >> 1);
            let ebits = k - 1;
            let msb2 = 2 + (rel & 1);
            (u64::from(msb2) << ebits) | (i & ((1u64 << ebits) - 1))
        };
        all_syms.push(val);
    }
    let prim_all = PrimitiveArray::from_iter(all_syms);

    for allow_flat in [false, true] {
        for mode in [PivcoTreeMode::Huffman, PivcoTreeMode::OrderPreserving] {
            // At 4,096 rows/block even worst-case 150-symbol trees fit in <= 16 KiB!
            let opts_4k = PivcoOptions {
                tree_mode: mode,
                pha_enabled: true,
                allow_flat,
                block_shift: 12,
            };
            let enc_4k = Pivco::encode_with_options(&prim_all, opts_4k, &mut ctx)?;
            let ws_4k = DecodeWorkspace::for_tree(enc_4k.as_view().tree(), 4_096);
            assert!(
                ws_4k.allocated_bytes() <= 16 * 1024,
                "expected 4K 150-sym DecodeWorkspace <= 16 KiB, got {} bytes",
                ws_4k.allocated_bytes()
            );
            assert_arrays_eq!(
                &enc_4k.into_array(),
                &prim_all.clone().into_array(),
                &mut ctx
            );

            // At 8,192 rows/block worst-case 150-symbol trees fit well inside 32 KiB L1D!
            let opts_8k = PivcoOptions {
                block_shift: 13,
                ..opts_4k
            };
            let enc_8k = Pivco::encode_with_options(&prim_all, opts_8k, &mut ctx)?;
            let ws_8k = DecodeWorkspace::for_tree(enc_8k.as_view().tree(), 8_192);
            assert!(
                ws_8k.allocated_bytes() <= 32 * 1024,
                "expected 8K 150-sym DecodeWorkspace <= 32 KiB, got {} bytes",
                ws_8k.allocated_bytes()
            );
            let arr = enc_8k.into_array();
            assert_arrays_eq!(&arr, &prim_all.clone().into_array(), &mut ctx);
            let rhs = ConstantArray::new(Scalar::from(16u64), prim_all.len()).into_array();
            assert_arrays_eq!(
                &arr.compare(rhs.clone(), CompareOperator::Lt)?,
                &prim_all
                    .clone()
                    .into_array()
                    .compare(rhs, CompareOperator::Lt)?,
                &mut ctx
            );
        }
    }
    Ok(())
}

// ============================================================================
// High-Value Regression Tests (P0 / P1 / P2 Hardening)
// ============================================================================

/// Regression 1a: Targeted `SPARSE` (duplicate/unsorted/OOB/dense), reserved rANS marker bits,
/// and truncated `raw_bits` corruptions must return `Err` without panicking.
#[test]
fn rejects_corrupt_sparse_rans_and_truncated_raw_bits() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut scratch = vec![MaybeUninit::<u8>::uninit(); 1024];

    for bad_sparse in [
        &[BM_MARKER_SPARSE_ONE, 0, 2, 0, 0, 0, 0, 0][..],
        &[BM_MARKER_SPARSE_ONE, 0, 2, 0, 5, 0, 2, 0][..],
        &[BM_MARKER_SPARSE_ONE, 0, 1, 0, 128, 0][..],
        &[BM_MARKER_SPARSE_ONE, 0, 3, 0, 0, 0, 1, 0, 2, 0][..],
    ] {
        assert!(NodeBitmap::read(bad_sparse, &mut 0, 128, &mut scratch).is_err());
    }
    for bad_marker in [BM_MARKER_RANS_BIT | 0x10, BM_MARKER_RANS_BIT | 0x20] {
        let mut wire = vec![bad_marker, 0, 128, 0];
        wire.extend(vec![0u8; 128]);
        assert!(NodeBitmap::read(&wire, &mut 0, 4096, &mut scratch).is_err());
    }

    let wide = PrimitiveArray::from_iter((0..1_024u64).map(|i| 1_000 + i));
    let wide_enc =
        Pivco::encode_with_options(&wide, PivcoOptions::pha().with_block_shift(10), &mut ctx)?
            .into_array();
    let wide_ser = PivcoPlugin
        .serialize(&wide_enc, &SESSION)?
        .vortex_expect("ser");
    let mut zero_raw_hdr =
        ByteBufferMut::with_capacity_aligned(wide_ser.buffers[0].len(), Alignment::of::<u32>());
    zero_raw_hdr.extend_from_slice(wide_ser.buffers[0].as_ref());
    zero_raw_hdr[12..16].copy_from_slice(&0u32.to_le_bytes());
    let zero_raw_bufs = vec![
        BufferHandle::new_host(zero_raw_hdr.freeze()),
        BufferHandle::new_host(wide_ser.buffers[1].clone()),
        BufferHandle::new_host(wide_ser.buffers[2].clone()),
    ];
    let zero_raw_arr = PivcoPlugin.deserialize(
        ArrayDeserialization::new(
            wide_ser.serialized_id,
            wide_enc.dtype(),
            wide_enc.len(),
            &wide_ser.metadata,
            &zero_raw_bufs,
            &wide_ser.children,
        ),
        &SESSION,
    )?;
    assert!(
        zero_raw_arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );
    assert!(zero_raw_arr.execute_scalar(0, &mut ctx).is_err());
    assert!(zero_raw_arr.execute_scalar(500, &mut ctx).is_err());
    Ok(())
}

fn check_corrupt_array_ops_never_panic_and_agree(
    corrupt_arr: &ArrayRef,
    filter_mask: &Mask,
    take_idx: &ArrayRef,
    cmp_rhs: &ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<bool> {
    let exec_res = corrupt_arr.clone().execute::<PrimitiveArray>(ctx);
    let zz_res = ZigZag::try_new(corrupt_arr.clone())?
        .into_array()
        .execute::<PrimitiveArray>(ctx);
    let filter_res = corrupt_arr
        .filter(filter_mask.clone())
        .and_then(|a| a.execute::<PrimitiveArray>(ctx));
    let take_res = corrupt_arr
        .take(take_idx.clone())
        .and_then(|a| a.execute::<PrimitiveArray>(ctx));
    let cmp_res = corrupt_arr
        .clone()
        .compare(cmp_rhs.clone(), CompareOperator::Lt)
        .and_then(|a| a.execute::<BoolArray>(ctx));

    let any_err = exec_res.is_err()
        || zz_res.is_err()
        || filter_res.is_err()
        || take_res.is_err()
        || cmp_res.is_err();

    if let Ok(exec_prim) = exec_res {
        let exec_arr = exec_prim.into_array();
        for spot in [0usize, 17, 1024, 4095, 4096, 4499] {
            if let Ok(sc) = corrupt_arr.execute_scalar(spot, ctx) {
                assert_eq!(sc, exec_arr.execute_scalar(spot, ctx)?);
            }
        }
        if let Ok(f_prim) = filter_res {
            assert_arrays_eq!(
                &f_prim.into_array(),
                &exec_arr.filter(filter_mask.clone())?,
                ctx
            );
        }
        if let Ok(t_prim) = take_res {
            assert_arrays_eq!(&t_prim.into_array(), &exec_arr.take(take_idx.clone())?, ctx);
        }
        if let Ok(c_bool) = cmp_res {
            assert_arrays_eq!(
                &c_bool.into_array(),
                &exec_arr.compare(cmp_rhs.clone(), CompareOperator::Lt)?,
                ctx
            );
        }
    }
    Ok(any_err)
}

/// Regression 1b: Seeded byte-flip mutations across `tree_payload` and `raw_bits` AFTER valid
/// `deserialize()` must NEVER panic across `execute()`, `ZigZag(Pivco)`, `scalar_at`, `filter`,
/// `take`, and `compare`, and whenever two paths both return `Ok`, they MUST agree.
#[test]
fn corrupt_payload_mutations_never_panic_and_agree() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0xC044_0971);

    let base = PrimitiveArray::from_iter((0..4_500u64).map(|i| {
        if i < 4_096 {
            let d = rng.random_range(0..1000u32);
            if d < 900 {
                0
            } else if d < 980 {
                1
            } else {
                100 + (i % 500)
            }
        } else {
            i % 200
        }
    }));
    let enc =
        Pivco::encode_with_options(&base, PivcoOptions::op_pha().with_block_shift(12), &mut ctx)?
            .into_array();
    let ser = PivcoPlugin.serialize(&enc, &SESSION)?.vortex_expect("ser");
    let filter_mask =
        Mask::from_buffer(BitBuffer::from_iter((0..base.len()).map(|i| i % 503 == 7)));
    let take_idx = PrimitiveArray::from_iter([0u32, 17, 1024, 4095, 4096, 4499]).into_array();
    let cmp_rhs = ConstantArray::new(Scalar::from(2u64), base.len()).into_array();

    let mut saw_err = false;
    for trial in 0..80usize {
        let mut tree_mut =
            ByteBufferMut::with_capacity_aligned(ser.buffers[1].len(), Alignment::of::<u32>());
        tree_mut.extend_from_slice(ser.buffers[1].as_ref());
        let mut raw_mut =
            ByteBufferMut::with_capacity_aligned(ser.buffers[2].len(), Alignment::of::<u64>());
        raw_mut.extend_from_slice(ser.buffers[2].as_ref());

        if trial < 60 && !tree_mut.is_empty() {
            let flips = 1 + (trial % 3);
            for _ in 0..flips {
                let pos = if trial < 10 {
                    trial % tree_mut.len()
                } else {
                    rng.random_range(0..tree_mut.len())
                };
                tree_mut[pos] ^= 1u8 << rng.random_range(0..8u32);
            }
        } else if !raw_mut.is_empty() {
            let pos = rng.random_range(0..raw_mut.len());
            raw_mut[pos] ^= 0xFF;
        }

        let bufs = vec![
            BufferHandle::new_host(ser.buffers[0].clone()),
            BufferHandle::new_host(tree_mut.freeze()),
            BufferHandle::new_host(raw_mut.freeze()),
        ];
        let Ok(corrupt_arr) = PivcoPlugin.deserialize(
            ArrayDeserialization::new(
                ser.serialized_id,
                enc.dtype(),
                enc.len(),
                &ser.metadata,
                &bufs,
                &ser.children,
            ),
            &SESSION,
        ) else {
            saw_err = true;
            continue;
        };

        if check_corrupt_array_ops_never_panic_and_agree(
            &corrupt_arr,
            &filter_mask,
            &take_idx,
            &cmp_rhs,
            &mut ctx,
        )? {
            saw_err = true;
        }
    }
    assert!(saw_err, "mutation trials must trigger corruption errors");
    Ok(())
}

/// Regression 2: `PivcoData::validate` / `PivcoPlugin::deserialize` must reject any `PivcoTree`
/// containing symbols whose max representable value exceeds `DType`'s non-negative range.
#[rstest]
#[case::i8_reject_36(PType::I8, 36, false)]
#[case::i8_accept_35(PType::I8, 35, true)]
#[case::u8_reject_38(PType::U8, 38, false)]
#[case::u8_reject_40(PType::U8, 40, false)]
#[case::u8_accept_37(PType::U8, 37, true)]
#[case::i16_reject_52(PType::I16, 52, false)]
#[case::i16_accept_51(PType::I16, 51, true)]
#[case::u16_reject_54(PType::U16, 54, false)]
#[case::u16_accept_53(PType::U16, 53, true)]
#[case::i32_reject_84(PType::I32, 84, false)]
#[case::i32_accept_83(PType::I32, 83, true)]
#[case::u32_reject_86(PType::U32, 86, false)]
#[case::u32_accept_85(PType::U32, 85, true)]
#[case::i64_reject_148(PType::I64, 148, false)]
#[case::i64_reject_149(PType::I64, 149, false)]
#[case::i64_accept_147(PType::I64, 147, true)]
#[case::u64_accept_149(PType::U64, 149, true)]
fn rejects_tree_symbols_exceeding_dtype_width(
    #[case] ptype: PType,
    #[case] sym: u8,
    #[case] expected_ok: bool,
) -> VortexResult<()> {
    let dtype = DType::Primitive(ptype, Nullability::NonNullable);
    let tree = Arc::new(PivcoTree::single_symbol(sym));
    let empty_hdr = BufferHandle::new_host(
        ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u32>()).freeze(),
    );
    let empty_tree = BufferHandle::new_host(
        ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u32>()).freeze(),
    );
    let empty_raw = BufferHandle::new_host(
        ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>()).freeze(),
    );

    let data = PivcoData::try_new(
        12,
        0,
        0,
        0,
        0,
        Arc::clone(&tree),
        empty_hdr.clone(),
        empty_tree.clone(),
        empty_raw.clone(),
    )?;
    let try_new_res = Pivco::try_new(dtype.clone(), 0, data, Validity::NonNullable);
    assert_eq!(
        try_new_res.is_ok(),
        expected_ok,
        "try_new ptype={ptype} sym={sym}"
    );

    let meta = PivcoMetadata {
        offset: 0,
        last_block_len: 0,
        block_shift: 12,
        tree_base_offset: 0,
        raw_base_word_offset: 0,
        tree_bytes: tree.to_bytes(),
    };
    let deser_res = PivcoPlugin.deserialize(
        ArrayDeserialization::new(
            crate::pivco_id(),
            &dtype,
            0,
            &meta.encode_to_vec(),
            &[empty_hdr, empty_tree, empty_raw],
            &[],
        ),
        &SESSION,
    );
    assert_eq!(
        deser_res.is_ok(),
        expected_ok,
        "deserialize ptype={ptype} sym={sym}"
    );
    Ok(())
}

/// Regression 3: Nullable `PrimitiveArray` with negative (`i64::MIN`, `-1`) or huge (`u64::MAX`)
/// garbage in masked-off null physical slots must encode cleanly without rejecting or polluting
/// extra-bits and must match canonical across `execute`, `scalar_at`, `filter`, `take`, and `compare`.
#[rstest]
#[case::ph(PivcoOptions::ph().with_block_shift(10))]
#[case::pha(PivcoOptions::pha().with_block_shift(10))]
#[case::op_pha(PivcoOptions::op_pha().with_block_shift(10))]
fn nullable_garbage_in_null_slots_matches_canonical(
    #[case] opts: PivcoOptions,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let n = 2_500usize;

    // Signed i64 with `i64::MIN` and `-1` in null physical slots, small non-negative in valid slots.
    let i64_vals: Buffer<i64> = (0..n)
        .map(|i| {
            if i % 5 == 0 {
                i64::MIN
            } else if i % 7 == 0 {
                -1
            } else {
                (i % 25) as i64
            }
        })
        .collect();
    let validity = Validity::from_iter((0..n).map(|i| i % 5 != 0 && i % 7 != 0));
    let dirty_i64 = PrimitiveArray::new(i64_vals, validity.clone());
    let enc_i64 = Pivco::encode_with_options(&dirty_i64, opts, &mut ctx)?;
    assert!(
        !enc_i64.as_view().has_extra_bits(),
        "masked-off i64::MIN/-1 garbage must not trigger extra bits"
    );
    let enc_i64_arr = enc_i64.into_array();
    let exp_i64_arr = dirty_i64.into_array();
    assert_arrays_eq!(&enc_i64_arr, &exp_i64_arr, &mut ctx);

    for idx in (0..n).step_by(11) {
        assert_eq!(
            enc_i64_arr.execute_scalar(idx, &mut ctx)?,
            exp_i64_arr.execute_scalar(idx, &mut ctx)?
        );
    }
    for c in [-1i64, 0, 1, 10, 24, 100] {
        let rhs = ConstantArray::new(Scalar::from(c), n).into_array();
        for op in [
            CompareOperator::Eq,
            CompareOperator::NotEq,
            CompareOperator::Lt,
            CompareOperator::Lte,
            CompareOperator::Gt,
            CompareOperator::Gte,
        ] {
            assert_arrays_eq!(
                &enc_i64_arr.clone().compare(rhs.clone(), op)?,
                &exp_i64_arr.clone().compare(rhs.clone(), op)?,
                &mut ctx
            );
        }
    }

    // Unsigned u64 with `u64::MAX` in null physical slots, `< 32` in valid slots.
    let u64_vals: Buffer<u64> = (0..n)
        .map(|i| {
            if i % 5 == 0 || i % 7 == 0 {
                u64::MAX
            } else {
                (i % 16) as u64
            }
        })
        .collect();
    let dirty_u64 = PrimitiveArray::new(u64_vals, validity);
    let enc_u64 = Pivco::encode_with_options(&dirty_u64, opts, &mut ctx)?;
    assert!(
        !enc_u64.as_view().has_extra_bits(),
        "masked-off u64::MAX garbage must not trigger extra bits"
    );
    let enc_u64_arr = enc_u64.into_array();
    let exp_u64_arr = dirty_u64.into_array();
    assert_arrays_eq!(&enc_u64_arr, &exp_u64_arr, &mut ctx);
    let mask = Mask::from_buffer(BitBuffer::from_iter((0..n).map(|i| i % 31 == 2)));
    assert_arrays_eq!(
        &enc_u64_arr.filter(mask.clone())?,
        &exp_u64_arr.filter(mask)?,
        &mut ctx
    );
    let take_idx = PrimitiveArray::from_iter([0u32, 5, 7, 100, 1024, 2499]).into_array();
    assert_arrays_eq!(
        &enc_u64_arr.take(take_idx.clone())?,
        &exp_u64_arr.take(take_idx)?,
        &mut ctx
    );
    Ok(())
}

// ============================================================================
// Tree Wire Format, Shared Cost Estimator, and Bernoulli rANS Unit Tests
// ============================================================================

#[test]
fn tree_wire_roundtrip_and_validation() -> VortexResult<()> {
    let mut counts = [0u64; MAX_SYMS];
    for (s, slot) in counts.iter_mut().enumerate() {
        *slot = 1u64 << ((MAX_SYMS - s).min(40));
    }

    for mode in [PivcoTreeMode::Huffman, PivcoTreeMode::OrderPreserving] {
        for allow_flat in [false, true] {
            let tree = PivcoTree::build(&counts, mode, allow_flat);
            let wire = tree.to_bytes();
            let decoded = PivcoTree::from_bytes(&wire)?;
            assert_eq!(decoded, tree);
            if mode == PivcoTreeMode::OrderPreserving {
                assert!(decoded.is_order_preserving());
            }
        }
    }

    assert!(PivcoTree::from_bytes(&[]).is_err());
    assert!(PivcoTree::from_bytes(&[0x00, 0x05, 0x00]).is_err());
    assert!(PivcoTree::from_bytes(&[0x01, 0x00, 0x03, 0x00, 0x03]).is_err());
    assert!(PivcoTree::from_bytes(&[0x00, u8::try_from(MAX_SYMS)?]).is_err());
    assert!(PivcoTree::from_bytes(&[0x07, 0x00]).is_err());
    let mut too_deep = vec![0x01; 12];
    for sym in 0..13u8 {
        too_deep.push(0x00);
        too_deep.push(sym);
    }
    assert!(PivcoTree::from_bytes(&too_deep).is_err());
    Ok(())
}

#[test]
fn estimate_compressed_bits_tracks_modes_and_validates_inputs() -> VortexResult<()> {
    let mut skewed = [0u64; MAX_SYMS];
    skewed[0] = 14_700;
    skewed[1] = 1_300;
    skewed[2] = 384;
    let total = 16_384usize;

    let (bits_ph, _) = estimate_compressed_bits(&skewed, total, 4_096, PivcoOptions::ph())?;
    let (bits_pha, _) = estimate_compressed_bits(&skewed, total, 4_096, PivcoOptions::pha())?;
    let (bits_op, is_op) = estimate_compressed_bits(&skewed, total, 4_096, PivcoOptions::op_pha())?;
    assert!(is_op);
    assert!(
        bits_pha < bits_ph,
        "PHA ({bits_pha}) should estimate fewer bits than PH ({bits_ph}) on 90% skewed stream"
    );
    assert!(bits_op > 0);

    // Null-fill: counts summing to < total_valid fills into best exact symbol `0..32`
    let (bits_with_nulls, _) =
        estimate_compressed_bits(&skewed, total + 1_000, 4_096, PivcoOptions::pha())?;
    assert!(bits_with_nulls >= bits_pha);

    // Empty input and invalid arguments
    assert_eq!(
        estimate_compressed_bits(&[0u64; MAX_SYMS], 0, 4_096, PivcoOptions::pha())?,
        (0, true)
    );
    assert!(estimate_compressed_bits(&skewed, total, 0, PivcoOptions::pha()).is_err());
    assert!(estimate_compressed_bits(&skewed, total - 1, 4_096, PivcoOptions::pha()).is_err());
    Ok(())
}

#[test]
fn exact_integer_bernoulli_tables_sum_to_2048_and_are_monotone_by_popcount() {
    for table_idx in 0..16u64 {
        let p_num = table_idx + 1;
        let freqs = compute_bernoulli_freqs(p_num);
        let sum: u32 = freqs.iter().map(|&f| u32::from(f)).sum();
        assert_eq!(sum, 2048, "table p={p_num}/64 sum mismatch");
        assert!(freqs.iter().all(|&f| f >= 1));
        assert!(freqs[0x00] > freqs[0xFF]);
    }
}

#[test]
fn bitmap_rans_32lane_roundtrip_and_integrity_checks() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(0x3208_7777);
    let mut rans_words = Vec::new();
    for &len in &[
        1usize, 7, 8, 63, 64, 255, 256, 512, 1024, 2047, 2048, 4096, 8192,
    ] {
        for &p_pct in &[0u32, 1, 5, 12, 25, 50, 88, 99, 100] {
            let nbytes = len.div_ceil(8);
            let mut bm = vec![0u8; nbytes];
            let mut k_right = 0usize;
            for idx in 0..len {
                if rng.random_range(0..100u32) < p_pct {
                    bm[idx >> 3] |= 1u8 << (idx & 7);
                    k_right += 1;
                }
            }
            let mut wire = Vec::new();
            write_node_bitmap(&bm, len, k_right, 0, true, &mut rans_words, &mut wire);

            let mut cursor = 0usize;
            let mut scratch = vec![MaybeUninit::<u8>::uninit(); nbytes];
            let decoded = NodeBitmap::read(&wire, &mut cursor, len, &mut scratch)?;
            assert_eq!(cursor, wire.len());
            assert_eq!(decoded.count1(&scratch, len), k_right);
            for spot in [0usize, len / 2, len - 1] {
                let expected_bit = ((bm[spot >> 3] >> (spot & 7)) & 1) != 0;
                assert_eq!(decoded.bit(&scratch, len, spot), expected_bit);
            }
            match decoded.into_dense(&mut scratch, len) {
                DenseBitmap::AllZero => assert_eq!(k_right, 0),
                DenseBitmap::AllOne => assert_eq!(k_right, len),
                DenseBitmap::Bytes(slice) => assert_eq!(slice, bm.as_slice()),
            }
        }
    }

    for table_idx in 0..16usize {
        for xor_flip in [false, true] {
            for &len in &[64usize, 511, 2048, 4096, 8192] {
                let nbytes = len.div_ceil(8);
                let bm = make_bitmap(len, BitmapPattern::SkewedLow, 500 + table_idx as u64);
                let states =
                    encode_static_byte_rans_into(&bm, len, xor_flip, table_idx, &mut rans_words);
                let mut payload = Vec::with_capacity(BM_LANES * 4 + rans_words.len() * 2);
                for s in states {
                    payload.extend_from_slice(&s.to_le_bytes());
                }
                for &w in rans_words.iter().rev() {
                    payload.extend_from_slice(&w.to_le_bytes());
                }
                let mut dec = vec![MaybeUninit::<u8>::uninit(); nbytes];
                let decoded =
                    decode_static_byte_rans(&payload, len, xor_flip, table_idx, &mut dec)?;
                assert_eq!(decoded, bm.as_slice());

                // Corrupting states, truncating words, or appending a trailing word must fail.
                let mut zero_state = payload.clone();
                zero_state[..4].fill(0);
                assert!(
                    decode_static_byte_rans(&zero_state, len, xor_flip, table_idx, &mut dec)
                        .is_err()
                );
                let mut max_states = payload.clone();
                max_states[..BM_LANES * 4].fill(0xFF);
                assert!(
                    decode_static_byte_rans(&max_states, len, xor_flip, table_idx, &mut dec)
                        .is_err()
                );
                if payload.len() >= BM_LANES * 4 + 2 {
                    assert!(
                        decode_static_byte_rans(
                            &payload[..payload.len() - 2],
                            len,
                            xor_flip,
                            table_idx,
                            &mut dec,
                        )
                        .is_err()
                    );
                }
                let mut trailing_word = payload;
                trailing_word.extend_from_slice(&0u16.to_le_bytes());
                assert!(
                    decode_static_byte_rans(&trailing_word, len, xor_flip, table_idx, &mut dec)
                        .is_err()
                );
            }
        }
    }
    Ok(())
}

// ============================================================================
// Exhaustive SIMD Differential Tests (`0..=513`, all `d in 2..=6`, edge slacks)
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum BitmapPattern {
    AllZero,
    AllOne,
    AlternatingAA,
    Alternating55,
    SkewedLow,
    SkewedHigh,
    Uniform,
}

fn make_bitmap(len: usize, pattern: BitmapPattern, seed: u64) -> Vec<u8> {
    let nbytes = len.div_ceil(8);
    let mut bm = vec![0u8; nbytes];
    let mut rng = StdRng::seed_from_u64(seed);
    match pattern {
        BitmapPattern::AllZero => {}
        BitmapPattern::AllOne => bm.fill(0xFF),
        BitmapPattern::AlternatingAA => bm.fill(0xAA),
        BitmapPattern::Alternating55 => bm.fill(0x55),
        BitmapPattern::SkewedLow => {
            for idx in 0..len {
                if rng.random_range(0..100u32) < 3 {
                    bm[idx >> 3] |= 1u8 << (idx & 7);
                }
            }
        }
        BitmapPattern::SkewedHigh => {
            for idx in 0..len {
                if rng.random_range(0..100u32) < 97 {
                    bm[idx >> 3] |= 1u8 << (idx & 7);
                }
            }
        }
        BitmapPattern::Uniform => {
            for byte in &mut bm {
                *byte = rng.random();
            }
        }
    }
    if (len & 7) != 0 && !bm.is_empty() {
        bm[nbytes - 1] &= (1u8 << (len & 7)) - 1;
    }
    bm
}

#[rstest]
#[case::all_zero(BitmapPattern::AllZero, MERGE_OVERREAD)]
#[case::all_one(BitmapPattern::AllOne, MERGE_OVERREAD)]
#[case::alt_aa(BitmapPattern::AlternatingAA, MERGE_OVERREAD)]
#[case::alt_55(BitmapPattern::Alternating55, MERGE_OVERREAD)]
#[case::skew_low(BitmapPattern::SkewedLow, MERGE_OVERREAD)]
#[case::skew_high(BitmapPattern::SkewedHigh, MERGE_OVERREAD)]
#[case::uniform_full_slack(BitmapPattern::Uniform, MERGE_OVERREAD)]
#[case::uniform_zero_slack(BitmapPattern::Uniform, 0)]
fn simd_merges_and_popcount_match_scalar(#[case] pattern: BitmapPattern, #[case] slack: usize) {
    let mut rng = StdRng::seed_from_u64(0x51AD_1234 ^ (slack as u64));
    for len in 0..=513usize {
        let bm = make_bitmap(len, pattern, 1000 + len as u64);
        let k_right_ref = popcount_bits_scalar(&bm, len);
        let k_right_simd = popcount_bits(&bm, len);
        assert_eq!(k_right_simd, k_right_ref, "popcount mismatch at len={len}");
        let k_left = len - k_right_ref;

        let mut left = vec![0u8; k_left + slack];
        let mut right = vec![0u8; k_right_ref + slack];
        for byte in &mut left[..k_left] {
            *byte = rng.random();
        }
        for byte in &mut right[..k_right_ref] {
            *byte = rng.random();
        }
        left[k_left..].fill(0xDE);
        right[k_right_ref..].fill(0xAD);

        let mut out_simd = vec![0xCCu8; len];
        let mut out_ref = vec![0xCCu8; len];

        merge_vec_vec(&bm, len, &left, &right, &mut out_simd);
        merge_vec_vec_scalar(&bm, len, &left, &right, &mut out_ref);
        assert_eq!(out_simd, out_ref, "merge_vec_vec mismatch at len={len}");

        // Verify in-place left and in-place right merges match scalar too!
        let mut buf_l = vec![0xEEu8; len + slack];
        buf_l[k_right_ref..k_right_ref + k_left].copy_from_slice(&left[..k_left]);
        merge_vec_vec_in_place_left(&bm, len, k_right_ref, &right, &mut buf_l);
        assert_eq!(
            &buf_l[..len],
            out_ref.as_slice(),
            "in_place_left mismatch at len={len}"
        );

        let mut buf_r = vec![0xEEu8; len + slack];
        buf_r[k_left..k_left + k_right_ref].copy_from_slice(&right[..k_right_ref]);
        merge_vec_vec_in_place_right(&bm, len, &left, k_left, &mut buf_r);
        assert_eq!(
            &buf_r[..len],
            out_ref.as_slice(),
            "in_place_right mismatch at len={len}"
        );

        merge_cst_vec(&bm, len, 0x37, &right, &mut out_simd);
        merge_cst_vec_scalar(&bm, len, 0x37, &right, &mut out_ref);
        assert_eq!(out_simd, out_ref, "merge_cst_vec mismatch at len={len}");
        let mut buf_cv = vec![0xEEu8; len + slack];
        buf_cv[k_left..k_left + k_right_ref].copy_from_slice(&right[..k_right_ref]);
        merge_cst_vec_in_place(&bm, len, 0x37, k_left, &mut buf_cv);
        assert_eq!(&buf_cv[..len], out_ref.as_slice());

        merge_vec_cst(&bm, len, &left, 0x91, &mut out_simd);
        merge_vec_cst_scalar(&bm, len, &left, 0x91, &mut out_ref);
        assert_eq!(out_simd, out_ref, "merge_vec_cst mismatch at len={len}");
        let mut buf_vc = vec![0xEEu8; len + slack];
        buf_vc[k_right_ref..k_right_ref + k_left].copy_from_slice(&left[..k_left]);
        merge_vec_cst_in_place(&bm, len, k_right_ref, 0x91, &mut buf_vc);
        assert_eq!(&buf_vc[..len], out_ref.as_slice());

        merge_cst_cst(&bm, len, 0x12, 0xE4, &mut out_simd);
        merge_cst_cst_scalar(&bm, len, 0x12, 0xE4, &mut out_ref);
        assert_eq!(out_simd, out_ref, "merge_cst_cst mismatch at len={len}");
    }
}

/// Regression 4: `pack_flat_codes` / `unpack_flat_codes` differential test across `d in 2..=6`
/// including high `base_rank = 140` exercising `u8` modular wrap in NEON/AVX2 packers.
#[rstest]
fn simd_flat_pack_and_unpack_match_scalar(
    #[values(2u8, 3, 4, 5, 6)] depth: u8,
    #[values(17u8, 140)] base_rank: u8,
) {
    let mut rng =
        StdRng::seed_from_u64(0xF1A7_0000 + u64::from(depth) * 256 + u64::from(base_rank));
    let num_codes = 1usize << depth;
    let mut c2s = [0u8; 64];
    for (code, slot) in c2s.iter_mut().enumerate().take(num_codes) {
        *slot = (code as u8).wrapping_mul(3).wrapping_add(5) % 150;
    }

    for len in 0..=513usize {
        let ranks: Vec<u8> = (0..len)
            .map(|_| base_rank.wrapping_add(rng.random_range(0..num_codes) as u8))
            .collect();

        let mut packed_simd = Vec::new();
        let mut packed_ref = Vec::new();
        pack_flat_codes(depth, &ranks, base_rank, &mut packed_simd);
        pack_flat_codes_scalar(depth, &ranks, base_rank, &mut packed_ref);
        assert_eq!(
            packed_simd, packed_ref,
            "pack_flat_codes mismatch at d={depth}, base={base_rank}, len={len}"
        );

        let mut out_simd = vec![0xAAu8; len];
        let mut out_ref = vec![0xAAu8; len];
        unpack_flat_codes(depth, len, &packed_ref, &c2s, &mut out_simd);
        unpack_flat_codes_scalar(depth, len, &packed_ref, &c2s, &mut out_ref);
        assert_eq!(
            out_simd, out_ref,
            "unpack_flat_codes mismatch at d={depth}, base={base_rank}, len={len}"
        );

        for (idx, &rank) in ranks.iter().enumerate() {
            assert_eq!(
                out_simd[idx],
                c2s[usize::from(rank.wrapping_sub(base_rank))]
            );
        }
    }
}

#[rstest]
#[case::pivot_0(0)]
#[case::pivot_1(1)]
#[case::pivot_16(16)]
#[case::pivot_127(127)]
#[case::pivot_128(128)]
#[case::pivot_255(255)]
fn simd_partition_ranks_matches_scalar(#[case] pivot: u8) {
    let mut rng = StdRng::seed_from_u64(0xBA47_0000 + u64::from(pivot));
    for len in 0..=513usize {
        let ranks: Vec<u8> = (0..len).map(|_| rng.random::<u8>()).collect();
        let nbytes = len.div_ceil(8);
        let mut bm_simd = vec![0xFFu8; nbytes];
        let mut bm_ref = vec![0xFFu8; nbytes];
        let mut left_simd = vec![0u8; len + 16];
        let mut right_simd = vec![0u8; len + 16];
        let mut left_ref = vec![0u8; len + 16];
        let mut right_ref = vec![0u8; len + 16];

        let rc_simd =
            partition_ranks_by_pivot(&ranks, pivot, &mut bm_simd, &mut left_simd, &mut right_simd);
        let rc_ref =
            partition_ranks_scalar(&ranks, pivot, &mut bm_ref, &mut left_ref, &mut right_ref);
        let lc_ref = len - rc_ref;

        assert_eq!(rc_simd, rc_ref);
        assert_eq!(bm_simd, bm_ref);
        assert_eq!(&left_simd[..lc_ref], &left_ref[..lc_ref]);
        assert_eq!(&right_simd[..rc_ref], &right_ref[..rc_ref]);
    }
}

#[test]
fn simd_pack_64_bytes01_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x6401);
    for _ in 0..256 {
        let mut chunk = [0u8; 64];
        for byte in &mut chunk {
            *byte = rng.random_range(0..=1u8);
        }
        assert_eq!(pack_64_bytes01(&chunk), pack_64_bytes01_scalar(&chunk));
    }
}

/// Exhaustively verifies the `X86_MERGE_SHUF0`/`X86_MERGE_SHUF1` PSHUFB-complement + OR tables
/// across all `65,536` 16-bit masks (runs on every architecture including `aarch64`).
#[test]
fn x86_merge_pshufb_tables_exhaustive_16bit() {
    let left: [u8; 16] = core::array::from_fn(|idx| 0x10 + (idx as u8));
    let right: [u8; 16] = core::array::from_fn(|idx| 0xA0 + (idx as u8));

    let pshufb = |src: &[u8; 16], ctrl: &[u8; 16]| -> [u8; 16] {
        let mut out = [0u8; 16];
        for lane in 0..16 {
            let idx = ctrl[lane];
            out[lane] = if (idx & 0x80) != 0 {
                0
            } else {
                src[usize::from(idx & 0x0F)]
            };
        }
        out
    };

    for mask16 in 0..=u16::MAX {
        let lo = usize::from(mask16 & 0xFF);
        let hi = usize::from(mask16 >> 8);
        let s0 = X86_MERGE_SHUF0.0[lo];
        let s1 = X86_MERGE_SHUF1.0[hi];
        let mut merged = [0u8; 16];
        for lane in 0..8 {
            merged[lane] = s0[lane];
            merged[lane + 8] = s0[lane + 8].wrapping_add(s1[lane]);
        }
        let inv: [u8; 16] = core::array::from_fn(|lane| merged[lane] ^ 0xFF);
        let rr = pshufb(&right, &merged);
        let rl = pshufb(&left, &inv);
        let actual: [u8; 16] = core::array::from_fn(|lane| rr[lane] | rl[lane]);

        let bm = mask16.to_le_bytes();
        let mut expected = [0u8; 16];
        merge_vec_vec_scalar(&bm, 16, &left, &right, &mut expected);
        assert_eq!(actual, expected);
    }
}
