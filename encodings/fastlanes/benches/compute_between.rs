// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::unwrap_used)]

use num_traits::NumCast;
use rand::RngExt;
use rand::rngs::StdRng;
use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;

fn main() {
    divan::main();
}

fn generate_primitive_array<T: NativePType + NumCast>(
    rng: &mut StdRng,
    len: usize,
) -> PrimitiveArray {
    (0..len)
        .map(|_| T::from_usize(rng.random_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>()
}

fn generate_bit_pack_primitive_array<T: NativePType + NumCast>(
    rng: &mut StdRng,
    len: usize,
) -> ArrayRef {
    let a = (0..len)
        .map(|_| T::from_usize(rng.random_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    bitpack_to_best_bit_width(&a).vortex_expect("").into_array()
}

fn generate_alp_bit_pack_primitive_array<T: NativePType + NumCast>(
    rng: &mut StdRng,
    len: usize,
) -> ArrayRef {
    let a = (0..len)
        .map(|_| T::from_usize(rng.random_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    let alp = alp_encode(
        a.as_view(),
        None,
        &mut LEGACY_SESSION.create_execution_ctx(),
    )
    .vortex_expect("");

    let encoded = alp.encoded().to_primitive();

    let bp = bitpack_to_best_bit_width(&encoded)
        .vortex_expect("")
        .into_array();
    ALP::new(bp, alp.exponents(), None).into_array()
}

const BENCH_ARGS: &[usize] = &[2 << 10, 2 << 13, 2 << 14];

mod primitive {
    use divan::Bencher;
    use num_traits::NumCast;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::generate_primitive_array;

    #[divan::bench(
        types = [i32, i64, u32, u64, f32, f64],
        args = BENCH_ARGS,
    )]
    fn old_raw_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                let gte = arr
                    .clone()
                    .into_array()
                    .binary(
                        ConstantArray::new(min, arr.len()).into_array(),
                        Operator::Gte,
                    )
                    .vortex_expect("");
                let lt = arr
                    .clone()
                    .into_array()
                    .binary(
                        ConstantArray::new(max, arr.len()).into_array(),
                        Operator::Lt,
                    )
                    .vortex_expect("");
                gte.binary(lt, Operator::And)
                    .vortex_expect("")
                    .execute::<RecursiveCanonical>(ctx)
            })
    }

    #[divan::bench(
        types = [i32, i64, u32, u64, f32, f64],
        args = BENCH_ARGS,
    )]
    fn new_raw_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                arr.clone()
                    .into_array()
                    .between(
                        ConstantArray::new(min, arr.len()).into_array(),
                        ConstantArray::new(max, arr.len()).into_array(),
                        BetweenOptions {
                            lower_strict: NonStrict,
                            upper_strict: NonStrict,
                        },
                    )
                    .unwrap()
                    .execute::<RecursiveCanonical>(ctx)
                    .unwrap()
            })
    }
}

mod bitpack {
    use divan::Bencher;
    use num_traits::NumCast;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::generate_bit_pack_primitive_array;

    #[divan::bench(
        types = [i16, i32, i64],
        args = BENCH_ARGS,
    )]
    fn old_bp_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                let gte = arr
                    .clone()
                    .binary(
                        ConstantArray::new(min, arr.len()).into_array(),
                        Operator::Gte,
                    )
                    .vortex_expect("");
                let lt = arr
                    .clone()
                    .binary(
                        ConstantArray::new(max, arr.len()).into_array(),
                        Operator::Lt,
                    )
                    .vortex_expect("");
                gte.binary(lt, Operator::And)
                    .unwrap()
                    .execute::<RecursiveCanonical>(ctx)
                    .unwrap()
            })
    }

    #[divan::bench(
        types = [i16, i32, i64],
        args = BENCH_ARGS,
    )]
    fn new_bp_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_bit_pack_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                arr.clone()
                    .between(
                        ConstantArray::new(min, arr.len()).into_array(),
                        ConstantArray::new(max, arr.len()).into_array(),
                        BetweenOptions {
                            lower_strict: NonStrict,
                            upper_strict: NonStrict,
                        },
                    )
                    .unwrap()
                    .execute::<RecursiveCanonical>(ctx)
                    .unwrap()
            })
    }
}

mod alp {
    use divan::Bencher;
    use num_traits::NumCast;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::generate_alp_bit_pack_primitive_array;

    #[divan::bench(
        types = [f32, f64],
        args = BENCH_ARGS,
    )]
    fn old_alp_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                let gte = arr
                    .clone()
                    .binary(
                        ConstantArray::new(min, arr.len()).into_array(),
                        Operator::Gte,
                    )
                    .vortex_expect("");
                let lt = arr
                    .clone()
                    .binary(
                        ConstantArray::new(max, arr.len()).into_array(),
                        Operator::Lt,
                    )
                    .vortex_expect("");
                gte.binary(lt, Operator::And)
                    .unwrap()
                    .execute::<RecursiveCanonical>(ctx)
                    .unwrap()
            })
    }

    #[divan::bench(
        types = [f32, f64],
        args = BENCH_ARGS,
    )]
    fn new_alp_prim_test_between<T>(bencher: Bencher, len: usize)
    where
        T: NumCast + NativePType,
        vortex_array::scalar::Scalar: From<T>,
    {
        let min = T::from_usize(5561).vortex_expect("");
        let max = T::from_usize(6032).vortex_expect("");
        let mut rng = StdRng::seed_from_u64(0);
        let arr = generate_alp_bit_pack_primitive_array::<T>(&mut rng, len);

        bencher
            .with_inputs(|| (&arr, LEGACY_SESSION.create_execution_ctx()))
            .bench_refs(|(arr, ctx)| {
                arr.clone()
                    .between(
                        ConstantArray::new(min, arr.len()).into_array(),
                        ConstantArray::new(max, arr.len()).into_array(),
                        BetweenOptions {
                            lower_strict: NonStrict,
                            upper_strict: NonStrict,
                        },
                    )
                    .unwrap()
                    .execute::<RecursiveCanonical>(ctx)
                    .unwrap()
            })
    }
}
