#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
use num_traits::NumCast;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_alp::{ALPArray, alp_encode};
use vortex_array::arrays::{ConstantArray, PrimitiveArray};
use vortex_array::compute::{Operator, compare};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_to_best_bit_width;

fn main() {
    divan::main();
}

fn generate_primitive_array<T: NativePType + NumCast + PartialOrd>(
    len: usize,
    max_value: usize,
) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0);
    (0..len)
        .map(|_| T::from_usize(rng.random_range(0..max_value)).vortex_expect(""))
        .collect::<PrimitiveArray>()
}

fn generate_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    len: usize,
    max_value: usize,
) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let a = (0..len)
        .map(|_| T::from_usize(rng.random_range(0..max_value)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    bitpack_to_best_bit_width(&a).vortex_expect("").into_array()
}

fn generate_alp_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    len: usize,
    max_value: usize,
) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let a = (0..len)
        .map(|_| T::from_usize(rng.random_range(0..max_value)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    let alp = alp_encode(&a).vortex_expect("");

    let encoded = alp.encoded().to_primitive().vortex_expect("");

    let bp = bitpack_to_best_bit_width(&encoded)
        .vortex_expect("")
        .into_array();
    ALPArray::try_new(bp, alp.exponents(), alp.patches().cloned())
        .vortex_expect("")
        .into_array()
}

const BENCH_ARGS: &[(usize, usize)] = &[
    (1024, 10),
    (1 << 12, 10),
    (1 << 12, 50),
    // (1 << 12, 100),
    (1 << 12, 100),
    // (1 << 12, 1_000),
    (1 << 14, 10),
    (1 << 14, 50),
    (1 << 14, 100),
    // (1 << 14, 1_000),
    // (1 << 16, 10),
    // (1 << 16, 50),
    // (1 << 16, 100),
];

#[divan::bench(
        types = [u8, u32, i16, u64, f32, f64],
        args = BENCH_ARGS,
    )]
fn raw_prim_test_compare<T>(bencher: Bencher, (len, max_value): (usize, usize))
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(243).vortex_expect("");
    let arr = generate_primitive_array::<T>(len, max_value);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect("")
    })
}

#[divan::bench(
    types = [u8, u32, i16, i64],
    args = BENCH_ARGS,
)]
fn bp_prim_test_between<T>(bencher: Bencher, (len, max_value): (usize, usize))
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(243).vortex_expect("");
    let arr = generate_bit_pack_primitive_array::<T>(len, max_value);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect("")
    })
}

#[divan::bench(
    types = [f32, f64],
    args = BENCH_ARGS,
)]
fn alp_prim_test_between<T>(bencher: Bencher, (len, max_value): (usize, usize))
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(243).vortex_expect("");
    let arr = generate_alp_bit_pack_primitive_array::<T>(len, max_value);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect("")
    })
}

// #[divan::bench(types=[u8, u16, u32, u64], consts = [2,3,5,7])]
// fn bitpacking_cmp_fused<T, const W: usize>(bencher: Bencher)
// where
//     T: BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive + Copy,
//     T: BitPacking + BitPackingCompare + Copy,
//     [(); 128 * W / size_of::<T>()]:,
// {
//     let value = T::from_usize(1).expect("");
//     let values = [T::from_usize(2).expect(""); 1024];
//     let mut packed = [T::zero(); 128 * W / size_of::<T>()];
//
//     unsafe { BitPacking::unchecked_pack(W, &values, &mut packed) };
//
//     let mut unpacked = [false; 1024];
//
//     bencher.bench_local(|| {
//         unsafe {
//             unchecked_unpack_cmp_impl(W, &packed, &mut unpacked, Operator::Gte, value);
//             // BitPackingCompare::unchecked_unpack_cmp(
//             //     W,
//             //     &packed,
//             //     &mut unpacked,
//             //     |a, b| a == b,
//             //     black_box(value),
//             // );
//             // black_box(unpacked)
//         };
//     });
// }
