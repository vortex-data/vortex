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
    (1 << 12, 50),
    (1 << 12, 100),
    (1 << 12, 1000),
    (1 << 12, 10_000),
    (1 << 14, 50),
    (1 << 14, 10_000),
    (1 << 16, 50),
    (1 << 16, 10_000),
];

#[divan::bench(
        types = [i32, i64, u32, u64, f32, f64],
        args = BENCH_ARGS,
    )]
fn raw_prim_test_compare<T>(bencher: Bencher, (len, max_value): (usize, usize))
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let arr = generate_primitive_array::<T>(len, max_value);
    // println!("{}", arr.to_array().tree_display());

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect("")
    })
}

#[divan::bench(
    types = [i16, i32, i64, u16, u32, u64],
    args = BENCH_ARGS,
)]
fn bp_prim_test_between<T>(bencher: Bencher, (len, max_value): (usize, usize))
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let arr = generate_bit_pack_primitive_array::<T>(len, max_value);
    // println!("{}", arr.to_array().tree_display());

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
    let min = T::from_usize(5561).vortex_expect("");
    let arr = generate_alp_bit_pack_primitive_array::<T>(len, max_value);
    // println!("{}", arr.to_array().tree_display());

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect("")
    })
}
