// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small enum dictionaries with byte codes. Run the same benchmark on the baseline and candidate
//! commits to include allocation, dispatch, and dictionary execution in both measurements.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const LENGTHS: &[usize] = &[64, 1_024, 65_536, 1_000_000];
const CARDINALITIES: &[usize] = &[2, 4, 8, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 192, 256];

fn run<T: NativePType, const CARDINALITY: usize>(bencher: Bencher, len: usize, skewed: bool) {
    // Non-identity values prevent a copy of the codes from looking like a correct lookup.
    let values = (0..CARDINALITY)
        .map(|code| {
            let byte = (code as u8).wrapping_mul(37).wrapping_add(129);
            let value = if T::PTYPE == PType::I8 {
                i16::from(byte as i8)
            } else {
                i16::from(byte)
            };
            T::from(value).unwrap()
        })
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(0);
    let codes = (0..len)
        .map(|_| {
            if skewed && rng.random_ratio(9, 10) {
                0
            } else {
                rng.random_range(0..CARDINALITY) as u8
            }
        })
        .collect::<Vec<_>>();
    let expected = codes
        .iter()
        .map(|&code| values[usize::from(code)])
        .collect::<Vec<_>>();
    let dict = DictArray::try_new(
        PrimitiveArray::from_iter(codes).into_array(),
        PrimitiveArray::from_iter(values).into_array(),
    )
    .unwrap()
    .into_array();

    let actual = dict
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .unwrap();
    assert_eq!(actual.as_slice::<T>(), expected);

    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(dict, mut ctx)| dict.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench(
    types = [u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64],
    args = LENGTHS,
    consts = CARDINALITIES
)]
fn uniform<T: NativePType, const CARDINALITY: usize>(bencher: Bencher, len: usize) {
    run::<T, CARDINALITY>(bencher, len, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(
    types = [u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64],
    args = LENGTHS,
    consts = CARDINALITIES
)]
fn skewed<T: NativePType, const CARDINALITY: usize>(bencher: Bencher, len: usize) {
    run::<T, CARDINALITY>(bencher, len, true);
}
