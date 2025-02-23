use divan::Bencher;
use num_traits::NumCast;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_alp::{alp_encode, ALPArray};
use vortex_array::arrays::{ConstantArray, PrimitiveArray};
use vortex_array::compute::StrictComparison::NonStrict;
use vortex_array::compute::{
    between, binary_boolean, compare, BetweenOptions, BinaryOperator, Operator,
};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_to_best_bit_width;

fn main() {
    divan::main();
}

fn generate_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> PrimitiveArray {
    (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>()
}

fn generate_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> ArrayRef {
    let a = (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    bitpack_to_best_bit_width(&a).vortex_expect("").into_array()
}

fn generate_alp_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> ArrayRef {
    let a = (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).vortex_expect(""))
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

const BENCH_ARGS: &[usize] = &[2 << 10, 2 << 13, 2 << 14];

#[divan::bench(
    types = [i32, i64, u32, u64, f32, f64],
    args = BENCH_ARGS,
)]
fn old_raw_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        binary_boolean(
            &compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect(""),
            &compare(&arr, &ConstantArray::new(max, arr.len()), Operator::Lt).vortex_expect(""),
            BinaryOperator::And,
        )
        .vortex_expect("")
    })
}

#[divan::bench(
    types = [i32, i64, u32, u64, f32, f64],
    args = BENCH_ARGS,
)]
fn new_raw_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        between(
            &arr,
            &ConstantArray::new(min, arr.len()),
            &ConstantArray::new(max, arr.len()),
            &BetweenOptions {
                lower_strict: NonStrict,
                upper_strict: NonStrict,
            },
        )
        .vortex_expect("")
    })
}

#[divan::bench(
    types = [i16, i32, i64],
    args = BENCH_ARGS,
)]
fn old_bp_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        binary_boolean(
            &compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect(""),
            &compare(&arr, &ConstantArray::new(max, arr.len()), Operator::Lt).vortex_expect(""),
            BinaryOperator::And,
        )
    })
}

#[divan::bench(
    types = [i16, i32, i64],
    args = BENCH_ARGS,
)]
fn new_bp_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        between(
            &arr,
            &ConstantArray::new(min, arr.len()),
            &ConstantArray::new(max, arr.len()),
            &BetweenOptions {
                lower_strict: NonStrict,
                upper_strict: NonStrict,
            },
        )
    })
}

#[divan::bench(
    types = [f32, f64],
    args = BENCH_ARGS,
)]
fn old_alp_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        binary_boolean(
            &compare(&arr, &ConstantArray::new(min, arr.len()), Operator::Gte).vortex_expect(""),
            &compare(&arr, &ConstantArray::new(max, arr.len()), Operator::Lt).vortex_expect(""),
            BinaryOperator::And,
        )
    })
}

#[divan::bench(
    types = [f32, f64],
    args = BENCH_ARGS,
)]
fn new_alp_prim_test_between<T>(bencher: Bencher, len: usize)
where
    T: NumCast + NativePType,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).vortex_expect("");
    let max = T::from_usize(6032).vortex_expect("");
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        between(
            &arr,
            &ConstantArray::new(min, arr.len()),
            &ConstantArray::new(max, arr.len()),
            &BetweenOptions {
                lower_strict: NonStrict,
                upper_strict: NonStrict,
            },
        )
    })
}
