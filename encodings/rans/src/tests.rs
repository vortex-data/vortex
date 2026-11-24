// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::BLOCK_VALUES;
use crate::Rans;
use crate::RansArrayExt;
use crate::RansPlugin;
use crate::codec::BASE_TABLE;
use crate::codec::EBITS_TABLE;
use crate::codec::PROB_SCALE;
use crate::codec::RECIP48;
use crate::codec::encode_pow2_bin;
use crate::zigzag_decode;
use crate::zigzag_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn pow2_bin_bijection_covers_extremes() {
    let mut values: Vec<u64> = (0..=256).collect();
    for k in 5..64u32 {
        let p = 1u64 << k;
        values.extend([
            p - 1,
            p,
            p + 1,
            p + (p >> 1) - 1,
            p + (p >> 1),
            p + (p >> 1) + 1,
        ]);
    }
    values.extend([u64::MAX - 2, u64::MAX - 1, u64::MAX]);

    for u in values {
        let (sym, ebits, raw) = encode_pow2_bin(u);
        let sym_idx = usize::from(sym);
        assert!(sym_idx < 150);
        assert_eq!(ebits, EBITS_TABLE[sym_idx]);
        if ebits == 0 {
            assert_eq!(raw, 0);
        } else {
            assert!(raw < (1u64 << ebits));
        }
        assert_eq!(BASE_TABLE[sym_idx] | raw, u, "bijection failed for u={u}");
    }
}

#[test]
fn zigzag_bijection_covers_signed_extremes() {
    for val in [
        0i64,
        -1,
        1,
        -2,
        2,
        i64::from(i8::MIN),
        i64::from(i8::MAX),
        i64::from(i16::MIN),
        i64::from(i16::MAX),
        i64::from(i32::MIN),
        i64::from(i32::MAX),
        i64::MIN + 1,
        i64::MAX - 1,
        i64::MAX,
        i64::MIN,
    ] {
        assert_eq!(zigzag_decode(zigzag_encode(val)), val);
    }
    assert_eq!(zigzag_encode(i64::MIN), u64::MAX);
}

#[test]
fn recip48_exact_division_across_all_frequencies() {
    let max_x = (1u32 << 31) - 1;
    for f in 1..=PROB_SCALE {
        let mul = RECIP48[f as usize];
        for x in [
            0u32,
            1,
            f - 1,
            f,
            f + 1,
            32_767,
            32_768,
            (f << 20).saturating_sub(1),
            max_x - 1,
            max_x,
        ] {
            let q = ((u128::from(x) * u128::from(mul)) >> 48) as u32;
            assert_eq!(q, x / f, "RECIP48 mismatch for x={x}, f={f}");
        }
    }
}

#[rstest]
#[case::empty(0)]
#[case::one(1)]
#[case::thirty_one(31)]
#[case::thirty_two(32)]
#[case::thirty_three(33)]
#[case::one_thousand(1000)]
#[case::exact_block(4096)]
#[case::block_plus_one(4097)]
#[case::multi_block(10_000)]
fn roundtrip_lengths_and_extra_bits_modes(#[case] len: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(42 + len as u64);

    // 1. Small alphabet (`< 32`, `has_extra_bits == false`)
    let gt_vals: Buffer<u64> = (0..len)
        .map(|_| {
            let draw = rng.random_range(0..100u32);
            if draw < 90 {
                0
            } else if draw < 98 {
                1
            } else {
                2
            }
        })
        .collect();
    let gt_prim = PrimitiveArray::new(gt_vals, Validity::NonNullable);
    let gt_rans = Rans::encode(&gt_prim, &mut ctx)?;
    assert!(!gt_rans.has_extra_bits());
    assert!(!gt_rans.signed_zigzag());
    assert_arrays_eq!(gt_rans, gt_prim, &mut ctx);

    // 2. Wide alphabet (`u64::MAX` included, `has_extra_bits == true`)
    let wide_vals: Buffer<u64> = (0..len)
        .map(|i| match i % 11 {
            0 => 0,
            1 => 60,
            2 => 900,
            3 => u64::MAX,
            4 => u64::MAX - 17,
            5 => (1u64 << 40) + 12345,
            _ => rng.random_range(0..5000),
        })
        .collect();
    let wide_prim = PrimitiveArray::new(wide_vals, Validity::NonNullable);
    let wide_rans = Rans::encode(&wide_prim, &mut ctx)?;
    if len >= 2 {
        assert!(wide_rans.has_extra_bits());
    }
    assert_arrays_eq!(wide_rans, wide_prim, &mut ctx);

    Ok(())
}

#[test]
fn roundtrip_all_eight_integer_ptypes() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 8_500usize;

    let u8_arr = PrimitiveArray::from_iter((0..len).map(|i| ((i * 7) % 251) as u8));
    assert_arrays_eq!(Rans::encode(&u8_arr, &mut ctx)?, u8_arr, &mut ctx);

    let u16_arr = PrimitiveArray::from_iter((0..len).map(|i| ((i * 31) % 60_000) as u16));
    assert_arrays_eq!(Rans::encode(&u16_arr, &mut ctx)?, u16_arr, &mut ctx);

    let u32_arr = PrimitiveArray::from_iter((0..len).map(|i| (i as u32).wrapping_mul(2654435761)));
    assert_arrays_eq!(Rans::encode(&u32_arr, &mut ctx)?, u32_arr, &mut ctx);

    let u64_arr = PrimitiveArray::from_iter(
        (0..len).map(|i| if i == 17 { u64::MAX } else { (i as u64) * 1000 }),
    );
    assert_arrays_eq!(Rans::encode(&u64_arr, &mut ctx)?, u64_arr, &mut ctx);

    // Non-negative signed integers must leave `signed_zigzag == false`.
    let i32_nonneg = PrimitiveArray::from_iter((0..len as i32).map(|i| i % 19));
    let i32_nonneg_rans = Rans::encode(&i32_nonneg, &mut ctx)?;
    assert!(!i32_nonneg_rans.signed_zigzag());
    assert_arrays_eq!(i32_nonneg_rans, i32_nonneg, &mut ctx);

    // Negative signed integers across i8, i16, i32, i64 (including MIN/MAX).
    let i8_arr = PrimitiveArray::from_iter((0..len).map(|i| match i {
        0 => i8::MIN,
        1 => i8::MAX,
        _ => ((i % 15) as i8) - 7,
    }));
    let i8_rans = Rans::encode(&i8_arr, &mut ctx)?;
    assert!(i8_rans.signed_zigzag());
    assert_arrays_eq!(i8_rans, i8_arr, &mut ctx);

    let i16_arr = PrimitiveArray::from_iter((0..len).map(|i| match i {
        0 => i16::MIN,
        1 => i16::MAX,
        _ => ((i % 200) as i16) - 100,
    }));
    assert_arrays_eq!(Rans::encode(&i16_arr, &mut ctx)?, i16_arr, &mut ctx);

    let i32_arr = PrimitiveArray::from_iter((0..len as i32).map(|i| match i {
        0 => i32::MIN,
        1 => i32::MAX,
        _ => (i % 50) - 25,
    }));
    assert_arrays_eq!(Rans::encode(&i32_arr, &mut ctx)?, i32_arr, &mut ctx);

    let i64_arr = PrimitiveArray::from_iter((0..len).map(|i| match i {
        0 => i64::MIN,
        1 => i64::MAX,
        2 => i64::MIN + 1,
        _ => ((i % 9) as i64) - 4,
    }));
    let i64_rans = Rans::encode(&i64_arr, &mut ctx)?;
    assert!(i64_rans.signed_zigzag());
    assert_arrays_eq!(i64_rans, i64_arr, &mut ctx);

    Ok(())
}

fn make_nullable_fixture(len: usize) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(77);
    PrimitiveArray::from_option_iter((0..len).map(|i| {
        if (4090..4200).contains(&i) || rng.random_range(0..100u32) < 21 {
            None
        } else {
            let d = rng.random_range(0..100u32);
            Some(if d < 85 {
                0i64
            } else if d < 95 {
                -1
            } else if d < 99 {
                600
            } else {
                i64::MIN
            })
        }
    }))
}

#[test]
fn nullable_slice_and_plugin_serde() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let prim = make_nullable_fixture(12_500);
    let rans = Rans::encode(&prim, &mut ctx)?;
    assert_arrays_eq!(&rans, &prim, &mut ctx);

    // AllInvalid array round-trip
    let all_null = PrimitiveArray::new(Buffer::<i32>::zeroed(5_000), Validity::AllInvalid);
    let all_null_rans = Rans::encode(&all_null, &mut ctx)?;
    assert_arrays_eq!(&all_null_rans, &all_null, &mut ctx);
    assert_eq!(
        all_null_rans.execute_scalar(2500, &mut ctx)?,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
    );

    let rans_ref = rans.into_array();
    let slice1 = SESSION
        .kernels()
        .reduce_parent(
            &rans_ref,
            &SliceArray::new(rans_ref.clone(), 3500..11_200).into_array(),
            0,
        )?
        .expect("SliceReduce must reduce Rans");
    assert!(slice1.is::<Rans>());
    assert_eq!(slice1.as_::<Rans>().offset(), 3500);
    assert_arrays_eq!(&slice1, prim.slice(3500..11_200)?, &mut ctx);

    let slice2 = SESSION
        .kernels()
        .reduce_parent(
            &slice1,
            &SliceArray::new(slice1.clone(), 700..6_800).into_array(),
            0,
        )?
        .expect("double SliceReduce must reduce Rans");
    assert!(slice2.is::<Rans>());
    assert_eq!(slice2.as_::<Rans>().offset(), (4200 - BLOCK_VALUES) as u16);
    let expected_slice2 = prim.slice(4200..10_300)?;
    assert_arrays_eq!(&slice2, &expected_slice2, &mut ctx);

    // Direct plugin serialize + deserialize of the doubly-sliced RansArray
    let validity_ready = {
        let v = slice2.as_::<Rans>();
        let val = v.rans_validity().execute_mask(v.len(), &mut ctx)?;
        let concrete_val = match val {
            Mask::AllTrue(_) => Validity::AllValid,
            Mask::AllFalse(_) => Validity::AllInvalid,
            Mask::Values(mv) => Validity::from(mv.bit_buffer().clone()),
        };
        Rans::try_new(v.dtype().clone(), v.len(), v.data().clone(), concrete_val)?.into_array()
    };
    let ser = RansPlugin
        .serialize(&validity_ready, &SESSION)?
        .expect("RansPlugin must serialize sliced RansArray");
    let ser_buffers: Vec<BufferHandle> = ser
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect();
    let deser = RansPlugin.deserialize(
        ArrayDeserialization::new(
            ser.serialized_id,
            validity_ready.dtype(),
            validity_ready.len(),
            &ser.metadata,
            &ser_buffers,
            &ser.children,
        ),
        &SESSION,
    )?;
    assert_arrays_eq!(deser, expected_slice2, &mut ctx);
    Ok(())
}

#[test]
fn scalar_at_take_and_filter_kernels() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(88);
    let prim = make_nullable_fixture(12_500);
    let rans_ref = Rans::encode(&prim, &mut ctx)?.into_array();
    let sliced = rans_ref.slice(4200..10_300)?.optimize_ctx(&SESSION)?;
    assert!(sliced.is::<Rans>());
    let expected = prim.slice(4200..10_300)?;

    for probe in [0, 1, 15, 31, 32, 100, 1000, 3000, sliced.len() - 1] {
        assert_eq!(
            sliced.execute_scalar(probe, &mut ctx)?,
            expected.execute_scalar(probe, &mut ctx)?,
            "scalar mismatch at probe {probe}"
        );
    }

    let sorted_idx = PrimitiveArray::from_iter([0u32, 5, 31, 32, 500, 4100, 6099]).into_array();
    assert_arrays_eq!(
        sliced.take(sorted_idx.clone())?,
        expected.take(sorted_idx)?,
        &mut ctx
    );

    let rand_idx: ArrayRef = PrimitiveArray::from_option_iter(
        (0..200).map(|i| (i % 13 != 0).then(|| rng.random_range(0..sliced.len() as u64))),
    )
    .into_array();
    assert_arrays_eq!(
        sliced.take(rand_idx.clone())?,
        expected.take(rand_idx)?,
        &mut ctx
    );

    let dense_idx: ArrayRef =
        PrimitiveArray::from_iter((0..2000).map(|_| rng.random_range(0..sliced.len() as u32)))
            .into_array();
    assert_arrays_eq!(
        sliced.take(dense_idx.clone())?,
        expected.take(dense_idx)?,
        &mut ctx
    );

    let sparse_mask = Mask::from_indices(sliced.len(), vec![4500, 4501, 5900, 6099]);
    assert_arrays_eq!(
        sliced.filter(sparse_mask.clone())?,
        expected.filter(sparse_mask)?,
        &mut ctx
    );

    let bitmap_mask = Mask::from_iter((0..sliced.len()).map(|i| (i * 17) % 53 == 0));
    let optimized_filtered = sliced.filter(bitmap_mask.clone())?.optimize_ctx(&SESSION)?;
    assert_arrays_eq!(optimized_filtered, expected.filter(bitmap_mask)?, &mut ctx);

    let small_rans = Rans::encode(
        &PrimitiveArray::from_option_iter([Some(0i32), None, Some(1), Some(-5), Some(200)]),
        &mut ctx,
    )?
    .into_array();
    test_filter_conformance(&small_rans, &mut ctx);

    Ok(())
}
