// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::prelude::*;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;

fn main() {
    divan::main();
}

const N: usize = 100_000;

#[divan::bench]
fn cast_u16_to_u32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    #[expect(clippy::cast_possible_truncation)]
    let arr = PrimitiveArray::from_option_iter((0..N).map(|i| {
        if rng.random_bool(0.5) {
            None
        } else {
            Some(i as u16)
        }
    }))
    .into_array();
    // Pre-compute min/max so values_fit_in is a cache hit during the benchmark.
    arr.statistics()
        .compute_all(
            &[Stat::Min, Stat::Max],
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .ok();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}
