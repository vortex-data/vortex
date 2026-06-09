// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use num_traits::NumCast;
use rand::RngExt;
use rand::rngs::StdRng;
use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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

    bitpack_to_best_bit_width(&a, &mut SESSION.create_execution_ctx())
        .vortex_expect("")
        .into_array()
}

fn generate_alp_bit_pack_primitive_array<T: NativePType + NumCast>(
    rng: &mut StdRng,
    len: usize,
) -> ArrayRef {
    let a = (0..len)
        .map(|_| T::from_usize(rng.random_range(0..10_000)).vortex_expect(""))
        .collect::<PrimitiveArray>();

    let mut ctx = SESSION.create_execution_ctx();
    let alp = alp_encode(a.as_view(), None, &mut ctx).vortex_expect("");

    let encoded = alp
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .vortex_expect("");

    let bp = bitpack_to_best_bit_width(&encoded, &mut ctx)
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
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::SESSION;
    use crate::generate_primitive_array;

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
            .with_inputs(|| (&arr, SESSION.create_execution_ctx()))
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
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::SESSION;
    use crate::generate_bit_pack_primitive_array;

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
            .with_inputs(|| (&arr, SESSION.create_execution_ctx()))
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
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
    use vortex_error::VortexExpect;

    use crate::BENCH_ARGS;
    use crate::SESSION;
    use crate::generate_alp_bit_pack_primitive_array;

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
            .with_inputs(|| (&arr, SESSION.create_execution_ctx()))
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
