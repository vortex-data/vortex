use divan::Bencher;
use num_traits::NumCast;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_alp::{alp_encode, ALPArray};
use vortex_array::array::{ConstantArray, PrimitiveArray};
use vortex_array::compute::{between, binary_boolean, compare, BinaryOperator, Operator};
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;

fn main() {
    divan::main();
}

fn generate_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> PrimitiveArray {
    (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).unwrap())
        .collect::<PrimitiveArray>()
}

fn generate_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> Array {
    let a = (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).unwrap())
        .collect::<PrimitiveArray>();

    bitpack_to_best_bit_width(a).unwrap().into_array()
}

fn generate_alp_bit_pack_primitive_array<T: NativePType + NumCast + PartialOrd>(
    rng: &mut StdRng,
    len: usize,
) -> Array {
    let a = (0..len)
        .map(|_| T::from_usize(rng.gen_range(0..10_000)).unwrap())
        .collect::<PrimitiveArray>();

    let alp = alp_encode(&a).unwrap();

    let encoded = alp.encoded().into_primitive().unwrap();

    let bp = bitpack_to_best_bit_width(encoded).unwrap().into_array();
    ALPArray::try_new(bp, alp.exponents(), alp.patches())
        .unwrap()
        .into_array()
}

const BENCH_ARGS: &[usize] = &[
    1_000,
    10_000,
    100_000,
    600_000,
    1_000_000,
    10_000_000,
    100_000_000,
    // 1_000_000_000,
];

#[divan::bench(
    types = [f64],
    args = BENCH_ARGS,
)]
fn old_two_raw_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min2 = T::from_usize(2).unwrap();
    let max2 = T::from_usize(400).unwrap();
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            binary_boolean(
                &binary_boolean(
                    &compare(&arr, ConstantArray::new(min2, arr.len()), Operator::Gt).unwrap(),
                    &compare(&arr, ConstantArray::new(max2, arr.len()), Operator::Lt).unwrap(),
                    BinaryOperator::And,
                )
                .unwrap(),
                &binary_boolean(
                    &compare(&arr, ConstantArray::new(min, arr.len()), Operator::Gte).unwrap(),
                    &compare(&arr, ConstantArray::new(max, arr.len()), Operator::Lt).unwrap(),
                    BinaryOperator::And,
                )
                .unwrap(),
                BinaryOperator::And,
            )
            .unwrap()
        })
}

#[divan::bench(
    types = [f64],
    args = BENCH_ARGS,
)]
fn old_raw_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            binary_boolean(
                &compare(&arr, ConstantArray::new(min, arr.len()), Operator::Gte).unwrap(),
                &compare(&arr, ConstantArray::new(max, arr.len()), Operator::Lt).unwrap(),
                BinaryOperator::And,
            )
            .unwrap()
        })
}

#[divan::bench(
    types = [f64],
    args = BENCH_ARGS,
)]
fn new_raw_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            between(
                &arr,
                ConstantArray::new(min, arr.len()),
                ConstantArray::new(max, arr.len()),
            )
            .unwrap()
        })
}

#[divan::bench(
    types = [i16, i32, i64],
    args = BENCH_ARGS,
)]
fn old_bp_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            binary_boolean(
                &compare(&arr, ConstantArray::new(min, arr.len()), Operator::Gte).unwrap(),
                &compare(&arr, ConstantArray::new(max, arr.len()), Operator::Lt).unwrap(),
                BinaryOperator::And,
            )
        })
}

#[divan::bench(
    types = [i16, i32, i64],
    args = BENCH_ARGS,
)]
fn new_bp_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            between(
                &arr,
                ConstantArray::new(min, arr.len()),
                ConstantArray::new(max, arr.len()),
            )
        })
}

#[divan::bench(
    types = [f32, f64],
    args = BENCH_ARGS,
)]
fn old_alp_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            binary_boolean(
                &compare(&arr, ConstantArray::new(min, arr.len()), Operator::Gte).unwrap(),
                &compare(&arr, ConstantArray::new(max, arr.len()), Operator::Lt).unwrap(),
                BinaryOperator::And,
            )
        })
}

#[divan::bench(
    types = [f32, f64],
    args = BENCH_ARGS,
)]
fn new_alp_prim_test_between<T: NativePType>(bencher: Bencher, len: usize)
where
    T: NumCast,
    vortex_scalar::Scalar: From<T>,
{
    let min = T::from_usize(5561).unwrap();
    let max = T::from_usize(6032).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

    // println!("{}", arr.tree_display());

    bencher
        .with_inputs(|| arr.clone())
        .bench_local_values(|arr| {
            between(
                &arr,
                ConstantArray::new(min, arr.len()),
                ConstantArray::new(max, arr.len()),
            )
        })
}
