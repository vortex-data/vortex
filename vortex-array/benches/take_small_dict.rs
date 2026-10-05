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
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const LENGTHS: &[usize] = &[64, 1_024, 65_536, 1_000_000];
const CARDINALITIES: &[u8] = &[2, 4, 8, 16, 32];

fn run<const CARDINALITY: u8>(bencher: Bencher, len: usize, skewed: bool) {
    // Non-identity values prevent a copy of the codes from looking like a correct lookup.
    let values = (0..CARDINALITY)
        .map(|code| code.wrapping_mul(37).wrapping_add(129))
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(0);
    let codes = (0..len)
        .map(|_| {
            if skewed && rng.random_ratio(9, 10) {
                0
            } else {
                rng.random_range(0..CARDINALITY)
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
    assert_eq!(actual.as_slice::<u8>(), expected);

    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(dict, mut ctx)| dict.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = LENGTHS, consts = CARDINALITIES)]
fn uniform<const CARDINALITY: u8>(bencher: Bencher, len: usize) {
    run::<CARDINALITY>(bencher, len, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = LENGTHS, consts = CARDINALITIES)]
fn skewed<const CARDINALITY: u8>(bencher: Bencher, len: usize) {
    run::<CARDINALITY>(bencher, len, true);
}
