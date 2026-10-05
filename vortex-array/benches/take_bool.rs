// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Boolean dictionary lookups, including the small u8 dictionary fast path.

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    divan::main();
}

#[divan::bench(args = [63, 64, 65, 65_536], consts = [4, 7, 25, 50, 64, 65])]
fn dictionary<const VALUES: u8>(bencher: Bencher, len: usize) {
    let values = BoolArray::from_iter((0..VALUES).map(|i| i % 3 == 0)).into_array();
    let mut rng = StdRng::seed_from_u64(91);
    let indices =
        PrimitiveArray::from_iter((0..len).map(|_| rng.random_range(0..VALUES))).into_array();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (indices.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(indices, mut ctx)| {
            values
                .take(indices)
                .vortex_expect("valid indices")
                .execute::<BoolArray>(&mut ctx)
                .vortex_expect("boolean dictionary executes")
        });
}
